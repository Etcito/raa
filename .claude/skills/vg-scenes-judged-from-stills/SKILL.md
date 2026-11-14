---
name: vg-scenes-judged-from-stills
description: Use when editing virtual-geometry example scenes, presets or their capture criteria: each capture is judged from one still image, so the scene must visibly show what its criterion names.
---

# Virtual-geometry captures are judged from single stills

Each preset capture is scored by a judge that sees only one image. Most failed render runs came from a mismatch between the scene and its criterion text, not from a renderer crash.

## Rules
- A criterion may only name things that one still image can show. Do not name motion, history, toggles between frames, or counts the camera cannot see. If a property cannot be seen (for example, that virtual casters are off), reword the criterion. Do not hope the judge infers it.
- Make the scene show what the criterion asks: objects in frame and not hidden behind each other, shadows falling on visible ground, debug colours actually present.
- For occlusion presets, frame a view where the second occlusion phase really has something to recover. Otherwise the debug colour never appears and the capture cannot pass.
- Check that the effect is real in code before you blame the scene. Occlusion that changes nothing (the same drawn clusters with occlusion on and off) and spheres missing after the second phase were both real code defects.
- Look at the retained image yourself before you re-run the judge. Re-running an unchanged scene fails the same way.
- Never loosen a criterion to name less than the feature must show. Fix the scene, the code, or a wrong wording.
