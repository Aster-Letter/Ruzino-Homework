# Setup NVAPI headers for Slang compilation
# This script copies NVAPI headers to the Slang include directory
# so that DXC can find them when compiling shaders with NVAPI support

$nvapi_source = "C:\Users\Pengfei\WorkSpace\Ruzino\source\Runtime\renderer\resources\nvapi"
$slang_include = "C:\Users\Pengfei\WorkSpace\Ruzino\SDK\slang\include"

Write-Host "Copying NVAPI headers from $nvapi_source to $slang_include"

Copy-Item "$nvapi_source\*.h" -Destination $slang_include -Force

Write-Host "NVAPI headers copied successfully"
Write-Host "Files copied:"
Get-ChildItem "$slang_include\nvHLSL*.h" | ForEach-Object { Write-Host "  - $($_.Name)" }
