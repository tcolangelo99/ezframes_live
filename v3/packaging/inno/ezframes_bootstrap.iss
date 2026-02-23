#ifndef MyAppVersion
  #define MyAppVersion "3.0.0"
#endif

[Setup]
AppName=EzFrames
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\EzFrames
DefaultGroupName=EzFrames
OutputDir=.
OutputBaseFilename=ezframes_v3_bootstrap_installer
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
Compression=lzma2
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=max
PrivilegesRequired=lowest
WizardStyle=modern
DisableDirPage=yes
DisableProgramGroupPage=yes
UsePreviousAppDir=no
SetupIconFile=..\..\runtime_bootstrap\assets\icons\launcher_icon.ico
UninstallDisplayIcon={app}\assets\icons\ezframes_icon.ico

[Files]
; Bootstrap payload only (thin installer). Explicit sources avoid shipping stale workspace test artifacts.
Source: "..\..\runtime_bootstrap\runtime\*"; DestDir: "{app}\runtime"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\..\runtime_bootstrap\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\..\runtime_bootstrap\assets\*"; DestDir: "{app}\assets"; Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
Source: "..\..\runtime_bootstrap\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
Source: "..\..\runtime_bootstrap\prereqs\*"; DestDir: "{app}\prereqs"; Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
Source: "..\..\runtime_bootstrap\bootstrap_launcher.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\runtime_bootstrap\EzFramesLauncher.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\runtime_bootstrap\README.txt"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Dirs]
Name: "{app}\logs"
Name: "{app}\state"
Name: "{app}\workspace"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Icons]
Name: "{group}\EzFrames"; Filename: "{app}\EzFramesLauncher.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\EzFrames"; Filename: "{app}\EzFramesLauncher.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Best-effort prerequisite install. If missing, attempt to install VC++ runtime.
Filename: "{app}\EzFramesLauncher.exe"; WorkingDir: "{app}"; Description: "Launch EzFrames"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsVCRedist: Boolean;
var
  Installed: Cardinal;
begin
  Result := True;
  if RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) then
  begin
    if Installed = 1 then
      Result := False;
  end;
end;

procedure InstallVCRedistIfNeeded();
var
  ExePath: string;
  ResultCode: Integer;
begin
  if not NeedsVCRedist() then
  begin
    Log('VC++ runtime already installed.');
    exit;
  end;

  ExePath := ExpandConstant('{app}\prereqs\vc_redist.x64.exe');
  if not FileExists(ExePath) then
  begin
    Log('VC++ redistributable not found in prereqs folder.');
    exit;
  end;

  WizardForm.StatusLabel.Caption := 'Installing Microsoft VC++ Runtime...';
  if Exec(ExePath, '/install /quiet /norestart', ExpandConstant('{app}\prereqs'), SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    Log(Format('vc_redist.x64.exe exited with code %d', [ResultCode]))
  else
    Log(Format('Failed to execute vc_redist.x64.exe. Error %d', [ResultCode]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    InstallVCRedistIfNeeded();
end;
