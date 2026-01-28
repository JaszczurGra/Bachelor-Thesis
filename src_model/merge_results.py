



validation_folder = 'visualizations/sweep_xrqywefs_grouchy_penguin_23-01-2026_00:43:35'
learining = 'visualizations/sweep_xrqywefs_'

models = [
    "sparkling-sweep-2_linear:128_layers:3",
    "upbeat-sweep-21_linear:64_layers:3",
    "ancient-sweep-22_linear:15_layers:3",
    "earthy-sweep-7_cubic:15_layers:3",
    "radiant-sweep-6_cubic:64_layers:3",
    "valiant-sweep-23_cubic:128_layers:3",
    "ruby-sweep-8_bspline:10_layers:3",
    "apricot-sweep-9_bspline:15_layers:3",
    "super-sweep-11_linear:128_layers:4",
    "cosmic-sweep-12_linear:64_layers:4",
    "true-sweep-13_linear:15_layers:4",
    "smooth-sweep-16_cubic:15_layers:4",
    "fine-sweep-15_cubic:64_layers:4",
    "copper-sweep-14_cubic:128_layers:4",
    "lyric-sweep-17_bspline:10_layers:4",
    "still-sweep-18_bspline:15_layers:4"
]

output_file = 'visualizations/merged_results'

categories = ['collision','path_length','curvature']
import os 
def merge_results(folder, models, output_file):
    
    values = {}
    for model in models:
        model_folder = os.path.join(folder, model)


        if 'Original' not in values:
            file_path = os.path.join(model_folder, f"metrics_original.csv")
            if not os.path.exists(file_path):
                continue
            values['Original'] = [[] for _ in range(len(categories))]
            with open(file_path, 'r') as f:
                cat = [0]*len(categories)
                lines = f.readlines()

                cat = [lines[0].strip().split(',').index(categories[i]) for i in range(len(categories))]

                for line in lines[1:]:
                    v  = line.strip().split(',')
                    for i in range(len(categories)):
                        values['Original'][i].append(float(v[cat[i]]))


        file_path = os.path.join(model_folder, f"metrics_model.csv")
        if not os.path.exists(file_path):
            continue
        values[model] = [[] for _ in range(len(categories))]
        with open(file_path, 'r') as f:
            cat = [0]*len(categories)
            lines = f.readlines()

            cat = [lines[0].strip().split(',').index(categories[i]) for i in range(len(categories))]

            for line in lines[1:]:
                v  = line.strip().split(',')
                for i in range(len(categories)):
                    values[model][i].append(float(v[cat[i]]))

    for c in range(len(categories)):
        with open(output_file+'_'+categories[c]+'.csv', 'w') as f:
            models_list = list(values.keys())
            f.write(','.join(k if k == 'Original' else k.split('_')[1] + '_' + k.split('_')[2] for k in models_list) +'\n')

            for i in range(len(values['Original'][c])):
                row = []
                for model in models_list:
                    row.append(str(values[model][c][i]))
                f.write(','.join(row)+'\n')


            




merge_results(validation_folder, models, output_file+"_validation")
merge_results(learining, models, output_file+'_whole')