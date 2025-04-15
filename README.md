# GCA_lite
A lite version of GCA base and everything almost encapsulated into classes

## A biref intro of GCA:

A class inherit structure:

```
GCAbase
 | 
 | -- GCA time series
 | -- GCA image generation 
 | -- ... 
```

## Overall paradigm

# Initialize models
N generators, e.g. [GRU LSTM, Transformer]  # 3 generator models
N discriminators, e.g. [CNND1, CNND2, CNND3]  # 3 discriminator models

Generators use past window size to predict next 1 (to N maybe will realize in the future version) timestamp.

Now following are the present code logic. (Please point out if there exists any faults)
``` 
FOR e in EPOCHS: 
  # Main training loop
  # 1. Individual pre-training
  for generator in generators:
      train(generator, loss_fn=MSE, Cross Entropy)  # Train each generator separately with MSE loss
      
  for discriminator in discriminators:
      train(discriminator, loss_fn=ClassificationLoss)  # Train each discriminator with classification loss (0: no change, 1: up, -1: down)

  while e % k ==0: 
    # 2. Intra-group evaluation and selection
    best_generator = evaluate_and_select_best(generators, validation_data)
    best_discriminator = evaluate_and_select_best(discriminators, validation_data)
      
    # 3. Intra-group knowledge distillation
    distill(best_generator, worst_generator)
     
    # 4. Cross-group competition
    FOR e0 in k0: 
      adversarial_train(best_generator, best_discriminator)
      if not converge: 
        break
```




# Today's Tasks(please omit)

## Interface Design & Optimization
- [ ] Discuss performance issues (point 1) with CC and propose solutions
- [ ] Add train/predict mode switching functionality (point 6)
- [x] Implement model weight saving capability (related to point 6)
- [x] Design model library display to show only model names (point 3)

## Model Training Improvements
- [x] Evaluate hyperparameters currently available (window size, batch size, learning rate, etc.)
- [ ] Consider adding automatic parameter suggestion (optimal epochs)
- [ ] Review loss functions and algorithm implementations for performance improvement

## Documentation & Specifications
- [x] Document supported data types (time-series with any sequential period)
- [ ] Clarify prediction cycle behavior in documentation (single vs multiple periods)
- [x] Update documentation for model import process (Python files in model directory)

## Model Library Management
- [x] Implement model search functionality for user-added models (point 4)
  - [x] Continue expanding built-in model library (point 5) 
  - so far as we set an easy init


## Evaluation Metrics
- [?] NEED CC
- [x] Verify all evaluation charts are being generated properly:
  - [x] Price fitting curves (train/test sets)
  - [x] MSE loss curves
  - [x] Cross-adversarial loss curves (N^2)
  - [x] Discriminator loss curves