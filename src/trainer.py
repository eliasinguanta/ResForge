import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from codecarbon import EmissionsTracker
import torch.nn.functional as F

#custom functions
import utils




class IncrementalTrainer:
    def __init__(self, epochs=70, learning_rate=0.01, batch_size=256, frozen_layers=0, pretrained=True, representative_memory=True, distillation=True):
        
        self.epochs_per_train = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size # defines how many samples are processed before backpropagation.
        self.representative_memory = representative_memory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

        # converter to tensor and normalize CIFAR-100 stats (hardcoded for that specific dataset)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load CIFAR-100 training and test datasets
        self.train_dataset = datasets.CIFAR100(root='/tmp/data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR100(root='/tmp/data', train=False, download=True, transform=self.transform)

        self.model = models.resnet18(pretrained=pretrained).to(self.device)  # Load pretrained ResNet-18

        frozen_layers = min(frozen_layers, 6)  # Limit freezing to 6 major layers
        layers = [self.model.conv1, self.model.bn1, self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]

        # Freeze layers for feature extraction
        if frozen_layers > 0:
            for layer in layers[:frozen_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.max_classes = 100  # Total classes in CIFAR-100
        self.new_classes = 10  # Newly added classes in each phase (we start with 10)
        self.num_classes = self.new_classes  # Current number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features  , self.num_classes).to(self.device)  # Replace final FC layer

        self.old_model = None  # Stored for distillation loss

        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            momentum=0.5,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1) # Reduces the learning rate with time
        self.criterion = nn.CrossEntropyLoss()  # Standard classification loss

        self.T = 2.0  # Temperature for distillation loss
        if distillation:
            self.alpha = 0.7  # Balance between CE and distillation loss
        else:
            self.alpha = 0.0
        self.exemplar_set = {}  # Exemplar memory for each class
        self.class_means = {}  # Mean feature vectors per class for k-NN

        self.train_tracker = EmissionsTracker(save_to_file=False)  # CO2 tracking for training
        self.test_tracker = EmissionsTracker(save_to_file=False)  # CO2 tracking for testing
        self.cum_train_emissions = 0.0  # Cumulative CO2 from training
        self.cum_test_emissions = 0.0  # Cumulative CO2 from testing
        
        self.epoch = 0  # Current epoch count for logging

    
    def _update_exemplar_set(self, num_exemplars_per_class=20):
        self.model.eval()

        # find got exemplar images for each new class
        for class_idx in range(self.num_classes - self.new_classes, self.num_classes):
            
            # Extract features for all images of the class
            class_indices = [i for i, (_, label) in enumerate(self.train_dataset) if label == class_idx]
            class_subset = Subset(self.train_dataset, class_indices)
            class_loader = DataLoader(class_subset, batch_size=64, shuffle=False)

            all_features = []
            image_indices = []

            with torch.no_grad():
                for i, (images, _) in enumerate(class_loader):
                    images = images.to(self.device)
                    feats = utils.extract_features(self.model, images)
                    all_features.append(feats.cpu())
                    image_indices.extend(class_indices[i * 64: i * 64 + images.size(0)])

            all_features = torch.cat(all_features, dim=0)
            all_features = F.normalize(all_features, dim=1)

            # Compute feature class mean
            class_mean = all_features.mean(dim=0)
            class_mean = F.normalize(class_mean, dim=0)

            selected = []
            exemplar_features = []

            # Select exemplars using herding
            for _ in range(min(num_exemplars_per_class, len(all_features))):
                if exemplar_features:
                    mean_so_far = torch.stack(exemplar_features).mean(dim=0)
                    mean_so_far = F.normalize(mean_so_far, dim=0)
                else:
                    mean_so_far = torch.zeros_like(class_mean)

                distances = (class_mean - all_features).norm(dim=1)
                for idx in selected:
                    distances[idx] = float('inf')  # Skip already selected indices

                best_idx = distances.argmin().item()  # Select the closest to the mean
                selected.append(best_idx)
                exemplar_features.append(all_features[best_idx])

            # Map feature indices back to dataset indices
            exemplar_indices = [image_indices[idx] for idx in selected]

            # Update the exemplar set
            if class_idx in self.exemplar_set:
                current_indices = set(self.exemplar_set[class_idx])
                new_indices = [idx for idx in exemplar_indices if idx not in current_indices]
                self.exemplar_set[class_idx].extend(new_indices)
            else:
                self.exemplar_set[class_idx] = exemplar_indices



    def _update_class_means(self):
        self.class_means = {}

        # for each class
        for class_idx, exemplar_indices in self.exemplar_set.items():
            exemplar_subset = torch.utils.data.Subset(self.train_dataset, exemplar_indices)
            exemplar_loader = torch.utils.data.DataLoader(
                exemplar_subset, 
                batch_size=len(exemplar_indices), 
                shuffle=False
            )
    
            with torch.no_grad():
                for images, _ in exemplar_loader:
                    images = images.to(self.device)

                    # Extract features
                    features = utils.extract_features(self.model, images)

                    # compute mean
                    self.class_means[class_idx] = F.normalize(features.mean(dim=0), dim=0)  # L2-Normalisierung




    def expand_model(self, new_num_classes):
        # we have to actually add classes
        if new_num_classes <= 0:
            raise ValueError("new_num_classes must be greater than 0.")

        # Store old model for distillation loss
        self.old_model = copy.deepcopy(self.model).eval().to(self.device)
        
        # freeze old model shouldnt be needed but maybe i do some mistakes 
        for param in self.old_model.parameters():
            param.requires_grad = False

        # Adjust the number of new classes if it would exceed the maximum
        if new_num_classes + self.num_classes > self.max_classes:
            new_num_classes = self.max_classes - self.num_classes

        # Expand fc layer
        new_fc = nn.Linear(self.model.fc.in_features, self.num_classes + new_num_classes).to(self.device)
        with torch.no_grad():
            new_fc.weight[:self.num_classes] = self.model.fc.weight.data
            new_fc.bias[:self.num_classes] = self.model.fc.bias.data
        self.model.fc = new_fc

        # Update class vars
        self.new_classes = new_num_classes
        self.num_classes += new_num_classes

        # Reinitialize the optimizer and learning rate scheduler
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            momentum=0.5,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)



    def _loss(self, images, labels):
        outputs = self.model(images)

        # cross-entropy loss (currently)
        ce_loss = self.criterion(outputs, labels)

        # Initialize distillation and L2 regularization losses
        kd_loss = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device) 

        # If an old model exists, compute distillation and L2 loss
        if self.old_model is not None:
            with torch.no_grad():
                # Get outputs from the frozen old model
                old_outputs = self.old_model(images)

            # Only use outputs of old classes for distillation
            old_class_count = self.num_classes - self.new_classes

            # distillation loss
            soft_targets = torch.softmax(old_outputs / self.T, dim=1)
            log_probs = torch.log_softmax(outputs[:, :old_class_count] / self.T, dim=1)            
            kd_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (self.T * self.T)

            # L2 regularization
            with torch.no_grad():
                old_feats = utils.extract_features(self.old_model, images)
            new_feats = utils.extract_features(self.model, images)
            l2_reg = F.mse_loss(new_feats, old_feats)

            # Combine cross-entropy loss, distillation loss and L2 regularization
            beta = 1.0
            loss = ce_loss * (1 - self.alpha) + self.alpha * kd_loss + beta * l2_reg
        else:
            # If no old model exists, we use only cross-entropy loss
            loss = ce_loss

        # Return total loss (and components for logging)
        return loss, ce_loss, kd_loss


    def _train_one_epoch(self, train_loader):
        # Start tracking carbon emissions for this epoch
        self.train_tracker.start()

        self.model.train()

        # Initialize vars
        running_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0
        all_preds = []
        all_labels = []

        # Iterate over the training data
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Reset gradients (for the new backward propagation)
            self.optimizer.zero_grad()

            # Compute total, cross entropy and distillation loss
            loss, ce_loss, kd_loss = self._loss(images, labels)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Accumulate losses for logging
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_kd_loss += kd_loss.item()

            # Predict class labels from model outputs for logging
            _, predicted = torch.max(self.model(images), 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Stop carbon tracking and update cumulative emissions
        train_emissions = self.train_tracker.stop()
        self.cum_train_emissions += train_emissions

        # Compute and return training metrics (accuracy, loss, emissions, ...)
        metrics = utils.get_training_metrics(self.epoch, all_labels, all_preds, running_loss, running_ce_loss, running_kd_loss, train_emissions, self.cum_train_emissions, len(train_loader))
        return metrics



    def train(self):
        # Update exemplar sets and means for the current classes bc we need it for the test
        self._update_exemplar_set()
        self._update_class_means()
    
        # Prepare data loaders for training and testing based on current dataset, classes, exemplars, batch size, and memory strategy
        train_loader, test_loader = utils.prepare_data(self.train_dataset, self.test_dataset, self.num_classes, self.new_classes, self.exemplar_set, self.batch_size, self.representative_memory)

        # store metrics for each epoch during training and testing for logging
        all_train_metrics = []
        all_test_metrics = []

        # Train for a fixed number of epochs
        for _ in range(self.epochs_per_train):
            # Perform one epoch of training and collect training metrics
            train_metrics = self._train_one_epoch(train_loader)
            # Evaluate the model on the test set and collect testing metrics
            test_metrics = self.test(test_loader)

            # Save the collected metrics for later analysis
            all_train_metrics.append(train_metrics)
            all_test_metrics.append(test_metrics)

            # Update the learning rate scheduler after each epoch
            self.scheduler.step()
            # Increment the epoch counter (for logging)
            self.epoch += 1

        # Return all metrics
        return all_train_metrics, all_test_metrics





    def test(self, test_loader):
        # Start tracking emissions
        self.test_tracker.start()

        self.model.eval()

        #initialize logging vars
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_labels_old = []
        all_preds_old = []
        all_labels_new = []
        all_preds_new = []

        # Disable gradient calculations for efficiency
        with torch.no_grad():
            # Iterate over batches
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Extract features from the model (excluding fc layer)
                features = utils.extract_features(self.model, images)

                # Prediction by nearest class mean
                preds = []
                for f in features:
                    distances = {cls: torch.norm(f - mean) for cls, mean in self.class_means.items()}
                    preds.append(min(distances, key=distances.get))
                preds = torch.tensor(preds).to(self.device)
                
                
                # Store logging infos
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

                # Separate labels and predictions into old and new classes for detailed analysis (also for logging)
                for p, l in zip(preds, labels):
                    if l.item() < self.num_classes - self.new_classes:
                        all_labels_old.append(l.item())
                        all_preds_old.append(p.item())
                    else:
                        all_labels_new.append(l.item())
                        all_preds_new.append(p.item())

        # Stop emissions tracking for testing and accumulate total emissions
        test_emissions = self.test_tracker.stop()
        self.cum_test_emissions += test_emissions

        # Calculate and return detailed test metrics using stored labels and predictions
        metrics = utils.get_test_metrics(self.epoch, all_labels, all_preds, all_labels_old, all_preds_old, all_labels_new, all_preds_new, test_emissions, self.cum_test_emissions, self.cum_train_emissions)
        return metrics

