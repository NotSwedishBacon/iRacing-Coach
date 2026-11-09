# Simucube Settings

### Reconstruction filter

* Higher reduces details but feels smoother.

### Dampening

* More reduces oscillation and softens the feeling.

### Torque bandwidth limit

* Lower numbers smooth out feel but you lose details.

### Friction

* Constant torque is added to increase resistance of turning.
* A higher number will add more friction - but harder to distinguish when the wheel loads up due to cornering.

### Inertia

* Higher numbers make the wheel keep moving in the direction it is moving.

### Static force reduction

* High number reduces force in sustained forces...like long corners.

### Slew rate limit

* It affects how fast the torque can change.
* Lower numbers will smooth out feel but reduce detail.

### Ultra-low latency model

* Reduces the latencies between wheel and pc.
* Higher numbers can reduce wheel oscillating.
* Using this can allow you to reduce dampening and friction and still reduce oscillation.

### Notch Filter

From the research I did on notch filters I think it is likely best to just leave this off as it only really affects  a narrow frequency.  Having said that, here is what I have found doing my research on Notch Filters.

#### Quick Summary:

turned off = less oscillation, smoother less notchy feeling

Turned on = more oscillation and some inconsistent feelings...like a dead zone.

Center Freq: Lower numbers is less oscillation

Attenuation dB : lower numbers is less oscillation

Q Factor: lower numbers is less oscillation but less connected and more rubbery delayed feeling

#### More Details:

Peak/notch filter is a filter that operates on frequency, so if your wheel tends to oscillate at a certain frequency if you let go of the wheel, you can make a notch on that frequency. Or if a certain rumble strip on the track is problematic and hits a resonance frequency that makes your whole rig vibrate at that frequency causing issues with your downstairs neighbours, then you can filter that out.

There is no best way… this filter is insanely hard to set, If you need it for oscillation control… Basically you have to set it while driving the car, Use a fairly deep attenuation and a reasonably wide range and then drive the car adjusting the frequency until it feels like you are reducing the area of the signal you want reduced… You can then narrow the Q factor to better pinpoint and adjust attenuation to where it just corrects rather than over cutting… One thing to note on this filter is that once set there will be a dip ANYTIME that frequency is used SO it can cause other issues in the way the FFB feels… i.e the feeling of a dead spot while turning through a corner… Or a lack of feel at certain times…

### Smooth out Feel

* Lower slew rate limit - like 0.5 - 1.5. I think this is better than using a higher reconstruction filter as it does not affect response times, just smoothness.
* High reconstruction filter like 5-6 - I think above 6 you lose too much detail.
* Increase dampening but can feel slow to react if above 17%. 10% is a good baseline
* Increase friction - 5-6%

### To add more detail

* Lower reconstruction filter.
* Reduce dampening.
* Reduce friction - 3-4%. 4% is a good baseline
* Reduce or turn off static force reduction. - I always have this off.
* High or turn off slew rate limit.

### To reduce oscillation

* Higher ultra low latency mode - like 20%.
* Higher dampening - I think anything above 17% otherwise slows response too much.
* Higher friction - don't go above 5% otherwise it is harder to feel the ffb in the corners.
* Lower reconstruction filter - like 1-3.
* Lower slew rate limit.

###
