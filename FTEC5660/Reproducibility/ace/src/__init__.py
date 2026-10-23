"""
ACE (Agent-Curator-Environment) System

A playbook-based learning framework with three main components:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and provides feedback
- Curator: Updates the playbook based on feedback

Usage:
    from src import ACE
    
    ace_system = ACE(
        generator_client=client,
        reflector_client=client,
        curator_client=client,
        generator_model="model-name",
        reflector_model="model-name",
        curator_model="model-name"
    )
    
    # Offline adaptation
    results = ace_system.run(
        mode='offline',
        train_samples=train_data,
        val_samples=val_data,
        test_samples=test_data,  # Optional
        data_processor=processor,
        config=config
    )

    # Online adaptation
    results = ace_system.run(
        mode='online',
        test_samples=test_data,
        data_processor=processor,
        config=config
    )

    # Evaluation only
    results = ace_system.run(
        mode='eval_only',
        test_samples=test_data,
        data_processor=processor,
        config=config
    )
"""

def __getattr__(name):
    # Configuration/data inspection must not load model clients or tokenizers.
    from importlib import import_module
    if name == 'ACE':
        value = import_module('.ace', __name__).ACE
    elif name == 'ACEBatch':
        value = import_module('.ace_batch', __name__).ACEBatch
    elif name in ('Generator', 'Reflector', 'Curator'):
        value = getattr(import_module('.core', __name__), name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value

__all__ = ['ACE', 'ACEBatch', 'Generator', 'Reflector', 'Curator']

__version__ = "1.0.0"
