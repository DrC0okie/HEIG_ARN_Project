import json
import os
import random

# Configuration des paramètres
themes = [
    "3024-night", "a11y-dark", "blackboard", "base16-dark", "base16-light", "cobalt", "duotone",
    "hopscotch", "lucario", "material", "monokai", "night-owl", "nord", "one-light", "yeti",
    "vscode", "solarized light", "verminal", "oceanic-next", "one-light"
]

font_families = [
    "Hack", "Anonymous Pro", "Cascadia Code", "Droid Sans Mono", "Fantasque Sans Mono",
    "Fira Code", "IBM Plex Mono", "Inconsolata", "JetBrains Mono", "Monoid",
    "Source Code Pro", "Space Mono", "Ubuntu Mono"
]

output_dir = "carbon_configs"
num_files = 100

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Générer les configurations
configurations = []

for theme in themes:
    for _ in range(5):
        config = {
            "paddingVertical": "0px",
            "paddingHorizontal": "0px",
            "backgroundImage": None,
            "backgroundImageSelection": None,
            "backgroundMode": "color",
            "backgroundColor": "rgba(0,0,0,1)",
            "dropShadow": False,
            "dropShadowOffsetY": "20px",
            "dropShadowBlurRadius": "68px",
            "theme": theme,
            "windowTheme": "none",
            "language": "text/x-c++src", # changer language pour "python" ou "haskell" 
            "fontFamily": random.choice(font_families),
            "fontSize": f"{random.randint(12, 16)}px",
            "lineHeight": f"{random.randint(100, 200)}%",
            "windowControls": False,
            "widthAdjustment": False,
            "lineNumbers": random.choice([True, False]),
            "firstLineNumber": 1,
            "exportSize": "2x",
            "watermark": False,
            "squaredImage": False,
            "hiddenCharacters": False,
            "name": "",
            "width": "680",
            "highlights": None
        }
        configurations.append(config)

# Sauvegarder les configurations en fichiers JSON
for i, config in enumerate(configurations):
    file_path = os.path.join(output_dir, f"{i + 1}.json")
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

print(f"Génération de {num_files} fichiers de configuration terminée.")
