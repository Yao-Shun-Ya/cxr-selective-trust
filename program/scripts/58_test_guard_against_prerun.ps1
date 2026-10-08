# Verify the strengthened [BAD] guard against the real pre-run artifacts.
#
# The guard is EXTRACTED FROM THE DRIVER rather than reimplemented, because a test
# that exercises a copy of the logic proves nothing about the shipped code -- that
# mistake was already made twice in this project (a stubbed test that loaded the
# real model, and an ECE convention check that used a non-discriminating vector).
#
# Expected: llava (empty replies) and medgemma_15 precision (unparsed thought
# traces) must be BAD; medgemma_4b shift must be OK.

$ErrorActionPreference = 'Continue'
$driver = Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\run_all_layers.ps1'
if (-not (Test-Path $driver)) { $driver = 'scripts\run_all_layers.ps1' }
$text = Get-Content $driver -Raw

function Extract-Fn([string] $name) {
    $m = [regex]::Match($text, "(?ms)^function $name\b.*?\n\}")
    if (-not $m.Success) { throw "cannot extract $name from the driver" }
    return $m.Value
}

$script:OutputRoot = (Resolve-Path 'outputs\pretest').Path
$LayerDef = @{ nih = @{ Dir = 'nih_eval' } }

Invoke-Expression (Extract-Fn 'Get-PredictionFile')
Invoke-Expression (Extract-Fn 'Test-StageOutput')

$cases = @(
    @{ model = 'llava_med_v1_5_7b';  stage = 'shift';     want = $false; expect = 0;    why = 'empty replies (0% parse)' },
    @{ model = 'llava_med_v1_5_7b';  stage = 'reliance';  want = $false; expect = 0;    why = 'empty replies' },
    @{ model = 'medgemma_15_4b_it';  stage = 'precision'; want = $false; expect = 0;    why = 'unparsed thought traces' },
    @{ model = 'medgemma_4b_it';     stage = 'shift';     want = $true;  expect = 1900; why = '100% usable, full coverage' },
    @{ model = 'internvl3_5_8b';     stage = 'shift';     want = $true;  expect = 1900; why = '100% usable, full coverage' },
    @{ model = 'qwen3_vl_8b_instruct'; stage = 'reliance'; want = $true; expect = 400;  why = '100% usable, full coverage' },
    # chexagent wrote only 8 of 1900 cells. It was caught by the exit code this time,
    # but a stage that wrote a fraction of its work AND exited 0 would have passed the
    # quality check (8 of 8 rows were usable). This case exercises the coverage check.
    @{ model = 'chexagent_8b';       stage = 'shift';     want = $false; expect = 1900; why = 'only 8 of 1900 rows' }
)

$fail = 0
Write-Host ""
Write-Host "  guard verdict (ok = usable rows within tolerance AND coverage >= 95%):"
foreach ($c in $cases) {
    $r = Test-StageOutput $c.model 'nih' $c.stage -Expected $c.expect
    $verdict = if ($r.ok) { 'OK  ' } else { 'BAD ' }
    $pass = ($r.ok -eq $c.want)
    if (-not $pass) { $fail++ }
    Write-Host ("  {0} {1}{2,-22} {3,-10} n={4,-5} bad={5,-5} err={6,-4} unparsed={7,-4} empty={8,-4} cover={9,-7} | {10}" -f `
        $(if ($pass) { 'PASS' } else { 'FAIL' }), $verdict, $c.model, $c.stage, $r.n, $r.bad, $r.err, $r.unparsed, $r.empty,
        $(if ($null -eq $r.cover) { 'n/a' } else { "$($r.cover)%" }), $c.why)
}
Write-Host ""
if ($fail) { Write-Host "  $fail case(s) FAILED -- the guard still misses what the pre-run exposed" -ForegroundColor Red }
else { Write-Host "  all cases pass: the guard now catches every failure mode the pre-run exposed" -ForegroundColor Green }
exit $fail
