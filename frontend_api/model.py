import torch
import torch.nn as nn
import torch.nn.functional as F

class FurnitureEncoder(nn.Module):
    def __init__(self, num_room_types, num_furniture_names, num_furniture_types):
        super(FurnitureEncoder, self).__init__()       
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.room_embedding = nn.Embedding(num_room_types, 3)
        self.furniture_names_embedding = nn.Embedding(num_furniture_names + 1, 15)  # +1 for the padding index
        self.furniture_types_embedding = nn.Embedding(num_furniture_types + 1, 300)  # +1 for the padding index
        
        # Adjust the input dimension for fc1 according to the concatenated tensor size
        self.fc1 = nn.Linear(49658, 1000)
        self.bn3 = nn.BatchNorm1d(1000)
        
        # Two separate fully connected layers for mu and log_var
        self.fc_mu = nn.Linear(1000, 300)
        self.fc_log_var = nn.Linear(1000, 300)
        
    def forward(self, image, room_features_tensor, room_type_tensor, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor):
        # Process images through conv layers
        x = self.pool(F.relu(self.bn1(self.conv1(image))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Get embeddings for furniture and room types
        room_embeddings = self.room_embedding(room_type_tensor)
        furniture_name_embeddings = self.furniture_names_embedding(furniture_names_tensor)
        furniture_type_embeddings = self.furniture_types_embedding(furniture_types_tensor)

        placed_name_embeddings = self.furniture_names_embedding(placed_furniture_tensor[:, :, 0].long())
        placed_type_embeddings = self.furniture_types_embedding(placed_furniture_tensor[:, :, 1].long())
        placed_furniture = placed_furniture_tensor[:, :, 2:]

        # Concatenate all features
        room_embeddings = room_embeddings.view(room_embeddings.size(0), -1)
        furniture_features_tensor = furniture_features_tensor.view(furniture_features_tensor.size(0), -1)
        furniture_name_embeddings = furniture_name_embeddings.view(furniture_name_embeddings.size(0), -1)
        furniture_type_embeddings = furniture_type_embeddings.view(furniture_type_embeddings.size(0), -1)
        placed_name_embeddings = placed_name_embeddings.view(placed_name_embeddings.size(0), -1)
        placed_type_embeddings = placed_type_embeddings.view(placed_type_embeddings.size(0), -1)
        placed_furniture = placed_furniture.reshape(placed_furniture.size(0), -1)
        
        x = torch.cat([x, room_features_tensor, room_embeddings, furniture_features_tensor, furniture_name_embeddings, furniture_type_embeddings, placed_name_embeddings, placed_type_embeddings, placed_furniture], dim=1)
        x = F.relu(self.bn3(self.fc1(x)))
        
        # Output mean and log_var for reparameterization
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mean, log_var

class FurnitureDecoder(nn.Module):
    def __init__(self, num_furniture_names, num_furniture_types, num_rot_classes=4, num_furniture=15):
        super(FurnitureDecoder, self).__init__()
        self.num_furniture_names = num_furniture_names
        self.num_furniture_types = num_furniture_types
        self.num_rot_classes = num_rot_classes
        self.num_furniture = num_furniture
        
        self.furniture_names_embedding = nn.Embedding(num_furniture_names + 1, 15)  # +1 for the padding index
        self.furniture_types_embedding = nn.Embedding(num_furniture_types + 1, 300)  # +1 for the padding index
        
        # Adjust the input dimension for fc1 according to the concatenated tensor size
        self.fc1 = nn.Linear(9825, 3000)
        self.bn1 = nn.BatchNorm1d(3000)
        
        # Output layer for all attributes of all furniture pieces
        self.fc2 = nn.Linear(3000, self.num_furniture * (num_furniture_names + num_furniture_types + num_rot_classes + 2))
        
    def forward(self, enc, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor):
        furniture_name_embeddings = self.furniture_names_embedding(furniture_names_tensor)
        furniture_type_embeddings = self.furniture_types_embedding(furniture_types_tensor)

        placed_name_embeddings = self.furniture_names_embedding(placed_furniture_tensor[:, :, 0].long())
        placed_type_embeddings = self.furniture_types_embedding(placed_furniture_tensor[:, :, 1].long())
        placed_furniture = placed_furniture_tensor[:, :, 2:]
        
        # Concatenate all features
        furniture_features_tensor = furniture_features_tensor.view(furniture_features_tensor.size(0), -1)
        furniture_name_embeddings = furniture_name_embeddings.view(furniture_name_embeddings.size(0), -1)
        furniture_type_embeddings = furniture_type_embeddings.view(furniture_type_embeddings.size(0), -1)

        placed_name_embeddings = placed_name_embeddings.view(placed_name_embeddings.size(0), -1)
        placed_type_embeddings = placed_type_embeddings.view(placed_type_embeddings.size(0), -1)
        placed_furniture = placed_furniture.reshape(placed_furniture.size(0), -1)
        
        x = torch.cat([enc, furniture_features_tensor, furniture_name_embeddings, furniture_type_embeddings, placed_name_embeddings, placed_type_embeddings, placed_furniture], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        outputs = self.fc2(x)
        
        # Reshape to separate each furniture piece's attributes
        outputs = outputs.view(-1, self.num_furniture, self.num_furniture_names + self.num_furniture_types + 2 + self.num_rot_classes)

        # Split the tensor into separate tensors for each attribute
        # Extract logits for furniture names, types, and rotations
        name_logits = outputs[:, :, :self.num_furniture_names]
        type_logits = outputs[:, :, self.num_furniture_names:self.num_furniture_names + self.num_furniture_types]
        
        # Extract and normalize coordinates using sigmoid to ensure they're in the range [0, 1]
        coords = torch.sigmoid(outputs[:, :, self.num_furniture_names + self.num_furniture_types:self.num_furniture_names + self.num_furniture_types + 2])
        
        # Extract rotation logits
        rot_logits = outputs[:, :, -self.num_rot_classes:]

        # Now `name_logits`, `type_logits`, `coords`, and `rot_logits` are separate tensors
        # Each tensor has dimensions: (batch_size, num_furniture, attribute_size)
        # where `attribute_size` varies depending on the attribute

        return name_logits, type_logits, coords, rot_logits
    
class FurniturePlacementModel(nn.Module):
    def __init__(self, num_room_types, num_furniture_names, num_furniture_types):
        super(FurniturePlacementModel, self).__init__()
        self.num_room_types = num_room_types
        self.num_furniture_names = num_furniture_names
        self.num_furniture_types = num_furniture_types
        
        self.encoder = FurnitureEncoder(num_room_types, num_furniture_names, num_furniture_types)
        self.decoder = FurnitureDecoder(num_furniture_names, num_furniture_types)
    
    def forward(self, image, room_features_tensor, room_type_tensor, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor, conf_threshold=0.9, deterministic=False):
        mean, log_var = self.encoder(image, room_features_tensor, room_type_tensor, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor)        
        z = self.reparameterize(mean, log_var)
        name_logits, type_logits, coords, rot_logits = self.decoder(z, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor)
        
        # Use this to generate samples
        if deterministic:
            # Calculate probabilities from logits
            name_probs = F.softmax(name_logits, dim=-1)
            type_probs = F.softmax(type_logits, dim=-1)

            # Calculate confidence scores by taking the max probability
            name_confidence = torch.max(name_probs, dim=-1)[0]
            type_confidence = torch.max(type_probs, dim=-1)[0]

            # Calculate overall confidence
            overall_confidence = (name_confidence + type_confidence) / 2

            # Determine which furniture items are above the threshold
            valid_mask = overall_confidence > conf_threshold

            # Filter based on the valid_mask
            filtered_name_logits = name_logits[valid_mask]
            filtered_type_logits = type_logits[valid_mask]
            filtered_coords = coords[valid_mask]
            filtered_rot_logits = rot_logits[valid_mask]

            # Use argmax to convert logits to class indices for valid entries
            filtered_name_classes = torch.argmax(filtered_name_logits, dim=-1)
            filtered_type_classes = torch.argmax(filtered_type_logits, dim=-1)
            filtered_rot_classes = torch.argmax(filtered_rot_logits, dim=-1)

            # Return the filtered and converted outputs
            return filtered_name_classes, filtered_type_classes, filtered_coords, filtered_rot_classes
        
        # Use this for the loss function
        else:
            return name_logits, type_logits, coords, rot_logits, mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)  # Sample epsilon from a standard normal distribution
        return mean + eps * std  # Scale and shift epsilon

    def compute_reconstruction_loss(self, name_logits, type_logits, coords, rot_logits, target, json_files):
        name_pad_idx = self.num_furniture_names
        type_pad_idx = self.num_furniture_types

        # Flatten the batch and num_furniture dimensions for name and type logits to match the target
        batch_size, num_furniture, num_names = name_logits.size()
        _, _, num_types = type_logits.size()
        _, _, num_rotations = rot_logits.size()

        # Flatten the tensors for CrossEntropyLoss
        name_logits_flat = name_logits.view(-1, num_names)  # [batch_size * num_furniture, num_furniture_names]
        type_logits_flat = type_logits.view(-1, num_types)  # [batch_size * num_furniture, num_furniture_types]
        rot_logits_flat = rot_logits.view(-1, num_rotations)  # [batch_size * num_furniture, num_rot_classes]

        # Flatten the target tensors as well
        target_names_flat = target[:, :, 0].view(-1).long()  # [batch_size * num_furniture]
        target_types_flat = target[:, :, 1].view(-1).long()  # [batch_size * num_furniture]
        target_rots_flat = target[:, :, 4].view(-1).long()  # [batch_size * num_furniture]

        # For coorL1Lossdinates, keep the batch and num_furniture dimensions
        target_coords = target[:, :, 2:4]  # [batch_size, num_furniture, 2]
        
        # Ensure coords tensor is reshaped correctly
        coords_pred = coords.view(-1, 2)
        coords_target = target_coords.view(-1, 2)

        # Loss functions
        name_loss_fn = nn.CrossEntropyLoss(ignore_index=name_pad_idx)
        type_loss_fn = nn.CrossEntropyLoss(ignore_index=type_pad_idx)
        rot_loss_fn = nn.CrossEntropyLoss()
        coord_loss_fn = nn.MSELoss()

        # Compute losses
        name_loss = name_loss_fn(name_logits_flat, target_names_flat)
        type_loss = type_loss_fn(type_logits_flat, target_types_flat)
        rot_loss = rot_loss_fn(rot_logits_flat, target_rots_flat)
        coord_loss = coord_loss_fn(coords_pred, coords_target)

        # Apply the regularization term
        threshold = 0.1
        dist_penalty_weight = 3.0
        distance = torch.norm(coords_pred - coords_target, p=2, dim=1)
        dist_penalty = torch.clamp(distance - threshold, min=0)
        dist_penalty_loss = dist_penalty_weight * torch.mean(dist_penalty)

        # Compute the penalty for furniture placement outside the room
        # out_penalty_weight = 3.0
        
        # batch_size, num_furniture, _ = coords.size()
        # out_penalty = torch.zeros(batch_size, num_furniture, device=coords.device)
        # for i in range(batch_size):
        #     json_file = json_files[i]
        #     with open(json_file, 'r') as f:
        #         state = json.load(f)
        #     xs = [s[0] for s in state['norm_room_corners']]
        #     ys = [s[1] for s in state['norm_room_corners']]

        #     for j in range(num_furniture):
        #         test_x = coords[i, j, 0].item()
        #         test_y = coords[i, j, 1].item()
        #         if is_outside(test_x, test_y, xs, ys):
        #             out_penalty[i, j] = 1.0

        # out_penalty = out_penalty.view(-1)
        # out_penalty_loss = out_penalty_weight * torch.mean(out_penalty)
        
        # Weighting factors for each loss component
        name_weight = 1.0
        type_weight = 1.0
        rot_weight = 3.0
        coord_weight = 4.0

        # Combined loss with weighting factors
        # total_loss = name_weight * name_loss + type_weight * type_loss + rot_weight * rot_loss + coord_weight * coord_loss + dist_penalty_loss + out_penalty_loss 
        total_loss = name_weight * name_loss + type_weight * type_loss + rot_weight * rot_loss + coord_weight * coord_loss + dist_penalty_loss 
        return total_loss
    
    def compute_loss(self, name_logits, type_logits, coords, rot_logits, mean, log_var, target, json_files):
        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(name_logits, type_logits, coords, rot_logits, target, json_files)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # Weight for the KL divergence term (can be adjusted)
        kl_weight = 0.01

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss