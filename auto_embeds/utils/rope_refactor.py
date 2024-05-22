# %%
from rope.base.project import Project

# Set up the project
project = Project("/workspace/auto_embeds")

# Load the module where `plot_difference` is defined
module = project.get_module("path/to/module.py")

# Example of how to use the project and module objects
print(f"Loaded module: {module}")

# Example of how to interact with the project
# This is a placeholder for where you might apply changes or refactor code
# project.do(change)

# Save the changes made to the project
# project.close()
