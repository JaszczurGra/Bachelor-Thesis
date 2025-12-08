# Bachelor-Thesis

## Running Scripts

### Parallel Path Generation
Generate paths using multiple threads:
```bash
python src/parallel_generation.py -n 12 -r 3 -t 120 --map maps/sleepy_panda_001 --vis --save
```
- `-n 12`: Use 12 threads
- `-r 3`: 3 runs per planner
- `-t 120`: Max 120 seconds runtime
- `--map`: Load map from folder
- `--vis`: Enable visualization
- `--save`: Save results

### Visualize Results
View saved path data:
```bash
python src/visualizer.py -d data/xd/ -n 3
```
- `-d`: Data directory path
- `-n 3`: Show 3 plots at once

### Generate Maps
Create random obstacle maps:
```bash
python src/map_generator.py -n 50
```
- `-n 50`: Generate 50 maps