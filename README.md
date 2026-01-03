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
Create size and turning radius maps 
```bash
 python map_creation/map_gen.py -s 10 -t 10 -o test1 
 ```
 - s - num of size maps 
 - t - num of turning maps 
 - o - output folder in maps 


### Run slurm
```bash
./run_combined_slurm.sh --save test_1 -n 8 -t 30 -r 100 --map maps/SlurmTest1/ 
```
- normal parameters for paralel_generation + save folder which is used to combine the data 
