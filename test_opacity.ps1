# Test script for opacity optimization
Write-Host "Testing opacity optimization..." -ForegroundColor Cyan

# Change to Release binaries directory
Set-Location "C:\Users\Pengfei\WorkSpace\Ruzino\Binaries\Release"

# Test parameters
$scene = ".\material_tests\shader_ball_Detailed_Ramshorn_Shells_Wall.usdc"
$renderNodes = ".\..\..\..\Ruzino\Assets\render_nodes_save.json"
$output = ".\light_field.png"
$width = 1000
$height = 1000
$samples = 8

Write-Host "Running headless_render.exe..." -ForegroundColor Yellow
Write-Host "Scene: $scene" -ForegroundColor Gray
Write-Host "Output: $output" -ForegroundColor Gray
Write-Host "Resolution: ${width}x${height}, Samples: $samples" -ForegroundColor Gray
Write-Host ""

# Run the renderer
.\headless_render.exe $scene $renderNodes $output $width $height $samples

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nRender completed successfully!" -ForegroundColor Green
    if (Test-Path $output) {
        Write-Host "Output saved to: $output" -ForegroundColor Green
    }
} else {
    Write-Host "`nRender failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
