# Register the zh_v2 crawler watchdog for the current user (no admin, no password stored; same style as the S10 fetch task).
#   powershell -ExecutionPolicy Bypass -File scripts\register_zh_v2_watchdog.ps1
# Task "R2AI-zh-v2-watchdog": user = current user, LogonType Interactive (runs while the user is logged on), RunLevel Limited.
# Triggers: at logon + every 5 minutes. WakeToRun, StartWhenAvailable, MultipleInstances=IgnoreNew, 10 minute limit per check.
# The watchdog restarts `python -m r2ai.zh_v2.crawl run` when it is dead or hung and writes out\runs\zh-v2-crawl\status.json
# (state ALIVE / DEAD / DONE / PAUSED / HOLD); `python -m r2ai.zh_v2.watchdog --status` exits 1 on DEAD.
# $root is the repo that holds .venv, data, state and out (the main checkout); the code comes from this worktree via PYTHONPATH.
$ErrorActionPreference = 'Stop'
$worktree = Split-Path -Parent $PSScriptRoot
$root = 'D:\GitHub\r2ai-stage3'
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Missing $python" }
$user = "$env:USERNAME"
$arg = "/c set PYTHONPATH=$worktree\src&& `"$python`" -B -m r2ai.zh_v2.watchdog"
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $arg -WorkingDirectory $root
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'R2AI-zh-v2-watchdog' -Action $action -Trigger @($logon, $repeat) -Settings $settings `
    -Principal $principal -Force | Out-Null
Get-ScheduledTask -TaskName 'R2AI-zh-v2-watchdog' | Format-List TaskName, State
