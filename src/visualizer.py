import matplotlib.pyplot as plt
import math

class Visualizer:
    def __init__(self, n_plots):

        plt.ion()
        self.n_plots = n_plots
        n_cols = math.ceil(math.sqrt(self.n_plots))
        n_rows = math.ceil(self.n_plots / n_cols)
        
        self.fig, axs = plt.subplots(n_rows, n_cols)#, figsize=(5 * n_cols, 5 * n_rows))
        self.fig.suptitle(f'OMPL Car Planning - Continuous', fontsize=16)
        # fig.set_facecolor('#2e2e2e')
        # Flatten axes
        if self.n_plots == 1:
            self.axs_flat = [axs]
        elif n_rows == 1 or n_cols == 1:
            self.axs_flat = axs.flatten() if hasattr(axs, 'flatten') else list(axs)
        else:
            self.axs_flat = axs.flatten()
        
        # Initialize plots
        for idx, ax in enumerate(self.axs_flat):
            ax.set_title(f'Planner {idx} - waiting...')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            # ax.set_facecolor('#aaaaaa')
            ax.grid(True, alpha=0.3)
            if idx >= n_plots:
                ax.set_visible(False)

        plt.tight_layout()
        plt.pause(0.5)

        self.last_timestamps = [0.0] * self.n_plots

    def update(self, result_list):
        for i in range(self.n_plots):
                result = result_list[i]

                if result is not None and result['timestamp'] > self.last_timestamps[i]:
                    self.last_timestamps[i] = result['timestamp']

                    ax = self.axs_flat[i]

                    ax.clear()
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
                    ax.grid(True, alpha=0.3)
                    
                    result['planner'].visualize(ax)
                    ax.set_title(f'Planner {i} - Run {result["run"]} - Solved:  {"Exact" if result["solved"] else "Approximate" if result["solved"] is not None else "No solution"}')
                    

                    handles, labels = ax.get_legend_handles_labels()
                    unique_labels = dict(zip(labels, handles))
                    self.fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')

                    
                    legend_text = "\n".join(f"{key}: {value:.2f}" for key, value in result['randomized_params'].items())
                    ax.text(0.02, -0.1, legend_text, transform=ax.transAxes, 
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))


                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

                    plt.pause(0.01)


    def close(self):
        pass
