"""Optimizer and learning rate scheduler factories."""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any, Optional
import logging


class OptimizerFactory:
    """Factory class for creating optimizers."""
    
    def __init__(self):
        """Initialize OptimizerFactory."""
        self.logger = logging.getLogger('pytorch_pipeline')
    
    def create_optimizer(
        self,
        model_parameters,
        optimizer_config: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer from configuration.
        
        Args:
            model_parameters: Model parameters to optimize
            optimizer_config: Optimizer configuration dictionary
            
        Returns:
            Configured optimizer
        """
        optimizer_type = optimizer_config.get('type', 'adam').lower()
        learning_rate = optimizer_config.get('learning_rate', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        
        self.logger.info(f"Creating {optimizer_type} optimizer with lr={learning_rate}, wd={weight_decay}")
        
        if optimizer_type == 'adam':
            beta1 = optimizer_config.get('beta1', 0.9)
            beta2 = optimizer_config.get('beta2', 0.999)
            eps = optimizer_config.get('eps', 1e-8)
            amsgrad = optimizer_config.get('amsgrad', False)
            
            optimizer = optim.Adam(
                model_parameters,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad
            )
        
        elif optimizer_type == 'adamw':
            beta1 = optimizer_config.get('beta1', 0.9)
            beta2 = optimizer_config.get('beta2', 0.999)
            eps = optimizer_config.get('eps', 1e-8)
            amsgrad = optimizer_config.get('amsgrad', False)
            
            optimizer = optim.AdamW(
                model_parameters,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad
            )
        
        elif optimizer_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.0)
            dampening = optimizer_config.get('dampening', 0.0)
            nesterov = optimizer_config.get('nesterov', False)
            
            optimizer = optim.SGD(
                model_parameters,
                lr=learning_rate,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=nesterov
            )
        
        elif optimizer_type == 'rmsprop':
            alpha = optimizer_config.get('alpha', 0.99)
            eps = optimizer_config.get('eps', 1e-8)
            momentum = optimizer_config.get('momentum', 0.0)
            centered = optimizer_config.get('centered', False)
            
            optimizer = optim.RMSprop(
                model_parameters,
                lr=learning_rate,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered
            )
        
        elif optimizer_type == 'adagrad':
            lr_decay = optimizer_config.get('lr_decay', 0.0)
            eps = optimizer_config.get('eps', 1e-10)
            
            optimizer = optim.Adagrad(
                model_parameters,
                lr=learning_rate,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
                eps=eps
            )
        
        elif optimizer_type == 'adadelta':
            rho = optimizer_config.get('rho', 0.9)
            eps = optimizer_config.get('eps', 1e-6)
            
            optimizer = optim.Adadelta(
                model_parameters,
                lr=learning_rate,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay
            )
        
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return optimizer
    
    def get_supported_optimizers(self) -> list:
        """Get list of supported optimizer types.
        
        Returns:
            List of supported optimizer names
        """
        return ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'adadelta']


class SchedulerFactory:
    """Factory class for creating learning rate schedulers."""
    
    def __init__(self):
        """Initialize SchedulerFactory."""
        self.logger = logging.getLogger('pytorch_pipeline')
    
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_config: Dict[str, Any]
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from configuration.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_config: Scheduler configuration dictionary
            
        Returns:
            Configured scheduler or None if no scheduler specified
        """
        if not scheduler_config or scheduler_config.get('type') == 'none':
            return None
        
        scheduler_type = scheduler_config.get('type', 'none').lower()
        
        self.logger.info(f"Creating {scheduler_type} scheduler")
        
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            
            scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )
        
        elif scheduler_type == 'multistep':
            milestones = scheduler_config.get('milestones', [30, 60, 90])
            gamma = scheduler_config.get('gamma', 0.1)
            
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma
            )
        
        elif scheduler_type == 'exponential':
            gamma = scheduler_config.get('gamma', 0.95)
            
            scheduler = lr_scheduler.ExponentialLR(
                optimizer,
                gamma=gamma
            )
        
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            eta_min = scheduler_config.get('eta_min', 0)
            
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        
        elif scheduler_type == 'cosine_warm_restarts':
            T_0 = scheduler_config.get('T_0', 10)
            T_mult = scheduler_config.get('T_mult', 1)
            eta_min = scheduler_config.get('eta_min', 0)
            
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
        
        elif scheduler_type == 'reduce_on_plateau':
            mode = scheduler_config.get('mode', 'min')
            factor = scheduler_config.get('factor', 0.1)
            patience = scheduler_config.get('patience', 10)
            threshold = scheduler_config.get('threshold', 1e-4)
            threshold_mode = scheduler_config.get('threshold_mode', 'rel')
            cooldown = scheduler_config.get('cooldown', 0)
            min_lr = scheduler_config.get('min_lr', 0)
            eps = scheduler_config.get('eps', 1e-8)
            
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                eps=eps
            )
        
        elif scheduler_type == 'cyclic':
            base_lr = scheduler_config.get('base_lr', 1e-4)
            max_lr = scheduler_config.get('max_lr', 1e-2)
            step_size_up = scheduler_config.get('step_size_up', 2000)
            step_size_down = scheduler_config.get('step_size_down', None)
            mode = scheduler_config.get('mode', 'triangular')
            gamma = scheduler_config.get('gamma', 1.0)
            scale_fn = scheduler_config.get('scale_fn', None)
            scale_mode = scheduler_config.get('scale_mode', 'cycle')
            cycle_momentum = scheduler_config.get('cycle_momentum', True)
            base_momentum = scheduler_config.get('base_momentum', 0.8)
            max_momentum = scheduler_config.get('max_momentum', 0.9)
            
            scheduler = lr_scheduler.CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                mode=mode,
                gamma=gamma,
                scale_fn=scale_fn,
                scale_mode=scale_mode,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum
            )
        
        elif scheduler_type == 'one_cycle':
            max_lr = scheduler_config.get('max_lr', 1e-2)
            total_steps = scheduler_config.get('total_steps', None)
            epochs = scheduler_config.get('epochs', None)
            steps_per_epoch = scheduler_config.get('steps_per_epoch', None)
            pct_start = scheduler_config.get('pct_start', 0.3)
            anneal_strategy = scheduler_config.get('anneal_strategy', 'cos')
            cycle_momentum = scheduler_config.get('cycle_momentum', True)
            base_momentum = scheduler_config.get('base_momentum', 0.85)
            max_momentum = scheduler_config.get('max_momentum', 0.95)
            div_factor = scheduler_config.get('div_factor', 25.0)
            final_div_factor = scheduler_config.get('final_div_factor', 1e4)
            
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
                div_factor=div_factor,
                final_div_factor=final_div_factor
            )
        
        elif scheduler_type == 'linear':
            start_factor = scheduler_config.get('start_factor', 1.0)
            end_factor = scheduler_config.get('end_factor', 0.0)
            total_iters = scheduler_config.get('total_iters', 100)
            
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return scheduler
    
    def get_supported_schedulers(self) -> list:
        """Get list of supported scheduler types.
        
        Returns:
            List of supported scheduler names
        """
        return [
            'none', 'step', 'multistep', 'exponential', 'cosine',
            'cosine_warm_restarts', 'reduce_on_plateau', 'cyclic',
            'one_cycle', 'linear'
        ]
    
    def is_step_based_scheduler(self, scheduler_type: str) -> bool:
        """Check if scheduler is step-based (called every batch) or epoch-based.
        
        Args:
            scheduler_type: Type of scheduler
            
        Returns:
            True if step-based, False if epoch-based
        """
        step_based = ['cyclic', 'one_cycle']
        return scheduler_type.lower() in step_based
    
    def requires_metric(self, scheduler_type: str) -> bool:
        """Check if scheduler requires a metric (like ReduceLROnPlateau).
        
        Args:
            scheduler_type: Type of scheduler
            
        Returns:
            True if requires metric, False otherwise
        """
        metric_based = ['reduce_on_plateau']
        return scheduler_type.lower() in metric_based


def create_optimizer_and_scheduler(
    model_parameters,
    optimizer_config: Dict[str, Any],
    scheduler_config: Optional[Dict[str, Any]] = None
) -> tuple:
    """Convenience function to create optimizer and scheduler.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_config: Optimizer configuration
        scheduler_config: Scheduler configuration (optional)
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer_factory = OptimizerFactory()
    scheduler_factory = SchedulerFactory()
    
    optimizer = optimizer_factory.create_optimizer(model_parameters, optimizer_config)
    scheduler = None
    
    if scheduler_config:
        scheduler = scheduler_factory.create_scheduler(optimizer, scheduler_config)
    
    return optimizer, scheduler 