class VariableContainer:
    def __init__(self):
        self.variables = {}

    def add(self, name, variable):
        self.variables[name] = variable

    def __getattr__(self, name):
        if name in self.variables:
            return self.variables[name]
        else:
            raise AttributeError(f"'VariableContainer' object has no attribute '{name}'")

    def __repr__(self):
        return f"VariableContainer({self.variables})"