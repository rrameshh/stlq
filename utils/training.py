import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter


class TrainingMetrics:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.running_loss = 0.0
        self.correct = 0
        self.total = 0
        self.batch_count = 0
    
    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor):
        self.running_loss += loss
        _, predicted = predictions.max(1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets).sum().item()
        self.batch_count += 1
    
    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.total if self.total > 0 else 0.0
    
    @property
    def avg_loss(self) -> float:
        return self.running_loss / self.batch_count if self.batch_count > 0 else 0.0

def train_epoch(
    model, 
    train_loader, 
    optimizer, 
    epoch: int, 
    switch_hook, 
    writer: SummaryWriter,
    log_interval: int = 100
) -> Dict[str, float]:
    model.train()
    metrics = TrainingMetrics()

    deit_loss_fn = None
    if hasattr(model, 'teacher_model') and model.teacher_model is not None:
        from models.vision.deit import DeiTLoss  # Import here to avoid circular imports
        deit_loss_fn = DeiTLoss(distillation_alpha=0.5, distillation_tau=3.0)
        print(f"Using DeiT distillation loss (alpha=0.5, tau=3.0)")
    
    # accumulated_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if isinstance(outputs, tuple) and len(outputs) == 3 and deit_loss_fn is not None:
            loss_dict = deit_loss_fn(outputs, targets)
            loss = loss_dict['total_loss']

            cls_logits, dist_logits, _ = outputs
            averaged_logits = (cls_logits + dist_logits) / 2
            outputs_for_metrics = averaged_logits
            
            if (i + 1) % log_interval == 0:
                print(f"  Distillation - Hard: {loss_dict['hard_loss']:.4f}, "
                      f"Soft: {loss_dict['soft_loss']:.4f}, "
                      f"Total: {loss_dict['total_loss']:.4f}")
        
        elif isinstance(outputs, tuple):
            if len(outputs) == 2:
                # Two outputs (e.g., cls_logits, dist_logits)
                cls_logits, dist_logits = outputs
                averaged_logits = (cls_logits + dist_logits) / 2
            elif len(outputs) >= 1:
                averaged_logits = outputs[0]
            else:
                raise ValueError(f"Unexpected output format: {len(outputs)} outputs")
                
            loss = F.cross_entropy(averaged_logits, targets)
            outputs_for_metrics = averaged_logits
        else:
            loss = F.cross_entropy(outputs, targets)
            outputs_for_metrics = outputs

        # loss = loss / 8
        # accumulated_loss += loss.item()

        
        
        loss.backward()
        optimizer.step()
        # if (i + 1) % 8 == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
            
        #     # Update metrics with accumulated loss
        #     metrics.update(accumulated_loss, outputs_for_metrics, targets)
        #     accumulated_loss = 0.0
        
        metrics.update(loss.item(), outputs_for_metrics, targets)
        
        iteration = epoch * len(train_loader) + i
        # if switch_hook.after_train_iter(iteration):
        #     print("Switched to activation quantization mode")
        
        if (i + 1) % log_interval == 0:
            _log_training_progress(epoch, i, len(train_loader), metrics, writer, iteration)
            metrics.reset()
    
    return {
        "loss": metrics.avg_loss,
        "accuracy": metrics.accuracy
    }
    


def validate(model, test_loader, epoch: int, writer: SummaryWriter) -> float:
    
    # Put model in eval mode but keep BatchNorm in train mode
    model.eval()
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.train()  # Force BN to stay in train mode
    test_loss = 0
    correct = 0
    total = 0
  
    with torch.no_grad():
        for inputs, targets in test_loader:
            device = getattr(model, 'device', next(model.parameters()).device)
            if hasattr(model, 'config') and hasattr(model.config, 'device'):
                device = model.config.device
                
            inputs, targets = inputs.to(device), targets.to(device)
            dequant_outputs = model(inputs)
            loss = F.cross_entropy(dequant_outputs, targets)
            test_loss += loss.item()
            _, predicted = dequant_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    
    accuracy = 100.0 * correct / total

    avg_loss = test_loss/len(test_loader)
    
    print(f'Validation - Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print(f"Correct predictions: {correct}/{total}")
        
    # Log to TensorBoard
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    return accuracy


def save_checkpoint(
    model, 
    optimizer, 
    accuracy: float, 
    epoch: int, 
    save_dir: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        accuracy: Current accuracy
        epoch: Current epoch
        save_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': accuracy,
        'epoch': epoch,
    }
    
    # Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # # Save regular checkpoint
    # checkpoint_path = f'{save_dir}/checkpoint_epoch{epoch+1}.pth'
    # torch.save(state, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = f'{save_dir}/best_model.pth'
        torch.save(state, best_path)
        print(f'Best model saved at epoch {epoch+1} with accuracy {accuracy:.2f}%')
    
    print(f'Checkpoint saved at epoch {epoch+1}')


def _log_training_progress(
    epoch: int, 
    batch_idx: int, 
    total_batches: int, 
    metrics: TrainingMetrics, 
    writer: SummaryWriter,
    iteration: int
) -> None:
    """Log training progress to console and TensorBoard"""
    print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}/{total_batches}], '
          f'Loss: {metrics.avg_loss:.4f}, Accuracy: {metrics.accuracy:.2f}%')
    
    # Log to TensorBoard
    writer.add_scalar('Training/Loss', metrics.avg_loss, iteration)
    writer.add_scalar('Training/Accuracy', metrics.accuracy, iteration)