# EzFrames v3 Parity Checklist

Status legend:
- `[x]` implemented in v3 new app path
- `[~]` partially implemented
- `[ ]` not implemented yet

## Core Workflow
- `[x]` Select video and output directory
- `[x]` Process video with motion filtering
- `[x]` Streaming ffmpeg encode (no full-frame buffering in RAM)
- `[x]` Motion detection options (`Absdiff`, `Optical Flow`)
- `[x]` Legacy logarithmic min-area slider semantics by method

## Cutie Integration
- `[x]` Native Cutie-lite process (`ezframes.cutie_lite`) without Qt interactive demo
- `[x]` Workspace mask directory contract (`workspace/<video>/binary_masks`)
- `[x]` `num_objects` flow wired
- `[~]` Legacy UX parity around Cutie launch/return states

## Camera Motion Handling
- `[x]` ORB homography gating
- `[x]` SIFT/ORB toggle + SIFT scale wiring
- `[x]` Homography debug image sequence output

## Output Options
- `[x]` ProRes output via ffmpeg pipeline
- `[x]` Debug contour output
- `[x]` Mask-constrained processing
- `[x]` Output image sequence mode parity
- `[x]` Output masked alpha video parity

## Playback / UI Controls
- `[x]` In-app player with next/prev/play parity
- `[ ]` Pop-out player parity
- `[~]` Full control-state parity and labels

## Interpolation (RIFE)
- `[x]` RIFE interpolation callable in v3 path
- `[~]` UI controls parity (factor/profile/progress behaviors)

## Workspace / Ops
- `[x]` Workspace root support
- `[~]` Workspace migration from legacy paths
- `[x]` Purge workspace UX parity

## Launcher / Runtime
- `[x]` Manifest-driven update path
- `[x]` Install state file
- `[x]` AWS auth retained with retries/timeouts
- `[x]` Default launch mode set to new app (`EZFRAMES_APP_MODE=new`)
- `[x]` Legacy mode only as fallback (`EZFRAMES_APP_MODE=legacy`)
