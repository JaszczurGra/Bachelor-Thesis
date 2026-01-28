import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

categories = ['collision','path_length','curvature']
category_y = ['Collision %','Path Length (m)','Curvature (1/m)']
category_scaling = [100,7.5,1]
datasets= ['validation', 'whole']  

for dataset in datasets:
    for category in categories:

        df = pd.read_csv(f'merged_results_{dataset}_{category}.csv')

        # 2. Reshape the data (Melting)
        # This transforms columns [original, model 1, model 2, model 3] 
        # into two columns: 'Model Type' and 'Value'
        df_melted = df.melt(var_name='Model', value_name='Score')
        df_melted['Score'] = df_melted['Score'] * category_scaling[categories.index(category)]
        
        
        
        # df_melted['Model'] = [v for v in  df_melted['Model']]
        models = [
            'Original',
            'linear:128_layers:3', 'linear:128_layers:4',
            'linear:64_layers:3', 'linear:64_layers:4',
            'linear:15_layers:3', 'linear:15_layers:4',
            'cubic:128_layers:3', 'cubic:128_layers:4',
            'cubic:64_layers:3', 'cubic:64_layers:4',
            'cubic:15_layers:3', 'cubic:15_layers:4',
            'bspline:15_layers:3', 'bspline:15_layers:4',
            'bspline:10_layers:3', 'bspline:10_layers:4'
        ]

        # models = df.columns.tolist()

        # models = [models[0]] + models[1::2] + models[2::2]
        df_melted['Model'] = pd.Categorical(df_melted['Model'], categories=models, ordered=True)



        plt.figure(figsize=(15, 5))
        sns.set_theme(style="whitegrid")

        ax = sns.boxplot(x='Model', y='Score', data=df_melted, palette='Set3', width=0.5,showfliers=False,hue='Model')

        labels = [models[0]] + [
            f"{m.split('_')[0].split(':')[0]}, {m.split('_')[0].split(':')[1]} units, {m.split('_')[1].split(':')[1]} layers"
            for m in models[1:]
        ]
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        # plt.title('Comparison of Model Performance', fontsize=15)
        plt.xlabel('', fontsize=12)
        plt.ylabel(category_y[categories.index(category)], fontsize=12)
        plt.tight_layout()
        plt.savefig(f'model_comparison_{dataset}_{category}.pdf', dpi=300)