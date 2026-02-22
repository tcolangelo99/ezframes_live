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

[Files]
; Bootstrap payload only (thin installer).
Source: "..\..\runtime_bootstrap\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\EzFrames"; Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\bootstrap_launcher.py"""; WorkingDir: "{app}"
Name: "{userdesktop}\EzFrames"; Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\bootstrap_launcher.py"""; WorkingDir: "{app}"

[Run]
; Best-effort prerequisite install. If missing, attempt to install VC++ runtime.
Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\bootstrap_launcher.py"""; WorkingDir: "{app}"; Description: "Launch EzFrames"; Flags: nowait postinstall skipifsilent

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
