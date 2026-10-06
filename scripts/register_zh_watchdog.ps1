# Register the full-zh crawler watchdog (run by the user, in an elevated PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\register_zh_watchdog.ps1
# - Allows wake timers on AC for every installed power scheme (Lenovo Vantage switches schemes).
# - Task "r2ai-zh-watchdog": at startup + every 5 minutes, runs whether the user is logged on or not,
#   wakes the computer, one instance at a time, 10 minute limit per check.
# The Windows password is typed by the user into the Get-Credential dialog; it is never stored in this repo.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Missing $python" }

# Wake timers on AC = Enable (RTCWAKE 1) for every scheme; the active one is re-applied.
$schemes = powercfg /list | Select-String -Pattern '([0-9a-f]{8}-[0-9a-f-]{27})' | ForEach-Object { $_.Matches[0].Value }
foreach ($g in $schemes) { powercfg /setacvalueindex $g SUB_SLEEP RTCWAKE 1 }
powercfg /setactive SCHEME_CURRENT

$action = New-ScheduledTaskAction -Execute $python -Argument '-B -m r2ai.zh_full.watchdog' -WorkingDirectory $root
$startup = New-ScheduledTaskTrigger -AtStartup
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$cred = Get-Credential -UserName "$env:USERDOMAIN\$env:USERNAME" -Message 'Windows password for r2ai-zh-watchdog (run whether logged on or not)'
Register-ScheduledTask -TaskName 'r2ai-zh-watchdog' -Action $action -Trigger @($startup, $repeat) -Settings $settings `
    -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited -Force | Out-Null

Get-ScheduledTask -TaskName 'r2ai-zh-watchdog' | Select-Object TaskName, State
foreach ($g in $schemes) { "$g RTCWAKE " + ((powercfg /query $g SUB_SLEEP RTCWAKE | Select-String 'Current AC').ToString().Trim()) }
