# -*- mode: python -*-

block_cipher = None


a = Analysis(['script.py'],
             pathex=['C:\\Users\\Yumraj-PC\\Desktop\\Ishan\\branch\\DeepInsightInstaller\\src\\di_ml_server'],
             binaries=[],
             datas=[],
             hiddenimports=['email.mime.multipart'],
             hookspath=[],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='script',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
