# %%
from rope.base.project import Project

# Set up the project
project = Project("/workspace/auto_embeds")

# Load the module where `plot_difference` is defined
module = project.get_module("path/to/module.py")
# Create a refactoring change to remove the 'query' argument
change = RemoveArgument(project, module, "plot_difference", arg_index=1)

# Preview the change before applying
description = change.get_description()
print(description)

# Apply the change
# project.do(change)

# Save the changes made to the project
# project.close()
