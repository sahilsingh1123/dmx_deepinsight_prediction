# -*- mode: python -*-

block_cipher = None


a = Analysis(['DIOptimus.py', 'start.py', 'ConfigureActions.py'],
             pathex=['/home/pallav/pallav/Optimus/OptimusSpark'],
             binaries=[],
             datas=[],
             hiddenimports=['datepaser'],
             hookspath=['.'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='DIOptimus',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='DIOptimus')
