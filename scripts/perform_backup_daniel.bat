@echo off

echo Renaming directory...
rename "C:\Users\danie\OneDrive\Skrivebord\backup" "backup_old"
echo Directory renamed.


echo Creating new directory...
mkdir C:\Users\danie\OneDrive\Skrivebord\backup
echo New directory created.

echo Transferring files...
scp -i C:\Users\danie\.ssh\daniel_hpc -r daen@hpc.itu.dk:/home/data_shares/scara/graphworld/results/ C:\Users\danie\OneDrive\Skrivebord\backup
echo Files transferred.
