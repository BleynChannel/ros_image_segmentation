if ($args[0] -eq "--rebuild") { Remove-Item -Force -Recurse -ErrorAction SilentlyContinue -Path $PSScriptRoot\build\}
New-Item -ItemType Directory -Force -Path $PSScriptRoot\build

Copy-Item -Path $PSScriptRoot\docker-compose.yml -Destination $PSScriptRoot\build

docker build -t bleyn/2d_segmentation:latest $PSScriptRoot\..
if (-not (Test-Path -Path $PSScriptRoot\build\2d_segmentation.tar)) {
	echo $PSScriptRoot\build\2d_segmentation.tar
	docker image save bleyn/2d_segmentation:latest -o $PSScriptRoot\build\2d_segmentation.tar
}

if (-not (Test-Path -Path $PSScriptRoot\build\data)) {
	New-Item -ItemType Directory -Force -Path $PSScriptRoot\build\data
	Copy-Item -Path $PSScriptRoot\..\data -Destination $PSScriptRoot\build\data\data\ -Recurse
}

'docker load --input $PSScriptRoot\2d_segmentation.tar
docker-compose -f $PSScriptRoot\docker-compose.yml up -d' | Out-File -FilePath $PSScriptRoot\build\deploy.ps1