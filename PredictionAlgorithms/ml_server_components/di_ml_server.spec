# -*- mode: python -*-

block_cipher = None


a = Analysis(['di_ml_server.py', 'entry.py', 'FPGrowth.py', 'KMeans.py', 'SentimentAnalysis.py', 'Forecasting.py'],
             pathex=['C:\\Users\\Yumraj-PC\\Desktop\\Ishan\\branch\\DeepInsightInstaller\\src\\di_ml_server'],
             binaries=[],
             datas=[],
             hiddenimports=[],
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
          name='di_ml_server',
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
               name='di_ml_server')
