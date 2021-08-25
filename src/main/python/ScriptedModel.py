
class ScriptedModel:
    def __init__(self, script):
        self.script = script

    def forward(self, cropped):
        return self.script(cropped)
