# GCA_lite
A lite version of GCA base and everything almost encapsulated into classes

# Today's Tasks

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