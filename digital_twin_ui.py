# digital_twin_ui.py
import ipywidgets as widgets
from IPython.display import display
from nibelungenbruecke.scripts.digital_twin_orchestrator.interface_test_DT import DigitalTwinInterface

def launch_ui(path):
    dt = DigitalTwinInterface(path)

    action_selector = widgets.Dropdown(
        options=['run', 'change', 'exit'],
        description='Action:',
        style={'description_width': 'initial'}
    )

    param_selector = widgets.Dropdown(
        options=['rho', 'E'],
        description='Parameter:',
        style={'description_width': 'initial'}
    )

    model_selector = widgets.Dropdown(
        options=dt.available_models,
        description='Model:',
        style={'description_width': 'initial'}
    )

    output = widgets.Output()
    run_button = widgets.Button(description="Submit", button_style='success')

    def on_button_clicked(b):
        output.clear_output()
        with output:
            action = action_selector.value
            if action == 'run':
                param = param_selector.value
                input_values = dt.generate_random_input(param)
                print(f"Generated input: {input_values}")
                try:
                    result = dt.run(input_values)
                    
                    print("Run complete.")
                    print("Result:", result)
                except Exception as e:
                    print("Error:", str(e))
            elif action == 'change':
                model = model_selector.value
                try:
                    dt.change_model(model)
                    print(f"Model changed to: {model}")
                except Exception as e:
                    print("Error:", str(e))
            elif action == 'exit':
                print("Exiting. (Notebook kernel continues running)")

    run_button.on_click(on_button_clicked)

    display(widgets.VBox([
        widgets.HTML("<h3>Digital Twin User Interface</h3>"),
        action_selector,
        param_selector,
        model_selector,
        run_button,
        date_picker
        output
    ]))
