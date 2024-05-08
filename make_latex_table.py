from src.utils import make_latex_table, color_table_by_method, color_table_by_method_type
if __name__ == "__main__":
    make_latex_table(micro=False, ML_metrics=False)
    # color_table_by_method(colors=['red', 'green', 'blue', 'yellow', 'purple', 'orange','brown' ])
    color_table_by_method_type()
    print("done")