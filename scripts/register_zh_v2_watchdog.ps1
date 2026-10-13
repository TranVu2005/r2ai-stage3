# Register the zh_v2 crawler watchdog (run by the user, in an elevated PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\register_zh_v2_watchdog.ps1
# Task "r2ai-zh-v2-watchdog": at startup + every 5 minutes, whether the user is logged on or not, wakes the computer, one instance at
# a time, 10 minute limit per check. The watchdog restarts `python -m r2ai.zh_v2.crawl run` when it is dead or hung and writes
# out\runs\zh-v2-crawl\status.json (state ALIVE / DEAD / DONE / PAUSED / HOLD); `python -m r2ai.zh_v2.watchdog --status` exits 1 on DEAD.
# The Windows password is typed by the user into the Get-Credential dialog; it is never stored in this repo.
# $root is the repo that holds .venv, data, state and out (the main checkout); the code comes from this worktree via PYTHONPATH.
$ErrorActionPreference = 'Stop'
$worktree = Split-Path -Parent $PSScriptRoot
$root = 'D:\GitHub\r2ai-stage3'
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Missing $python" }
$arg = "/c set PYTHONPATH=$worktree\src&& `"$python`" -B -m r2ai.zh_v2.watchdog"
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $arg -WorkingDirectory $root
$startup = New-ScheduledTaskTrigger -AtStartup
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$cred = Get-Credential -UserName "$env:USERDOMAIN\$env:USERNAME" -Message 'Windows password for r2ai-zh-v2-watchdog (run whether logged on or not)'
Register-ScheduledTask -TaskName 'r2ai-zh-v2-watchdog' -Action $action -Trigger @($startup, $repeat) -Settings $settings `
    -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited -Force | Out-Null
Get-ScheduledTask -TaskName 'r2ai-zh-v2-watchdog' | Select-Object TaskName, State
