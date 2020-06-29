pyinstaller --clean --additional-hooks-dir=. --hidden-import=datepaser DIOptimus.py start.py ConfigureActions.py
cp  -r pkgs/* dist/DIOptimus/