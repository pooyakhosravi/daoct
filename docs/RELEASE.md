# Public release checklist

- [x] Isolate the exact V14 submission without experimental variants or datasets.
- [x] Document method, entry points and known limitations.
- [x] Add BSD 2-Clause license text for project-owned code.
- [ ] Confirm author names, copyright holders and affiliations.
- [ ] Audit borrowed code and retain required starting-kit/upstream notices.
- [x] Record the final score report separately from earlier runs in the README.
- [ ] Request secure-server trained weights, model metadata and the optional anchor.
- [ ] Confirm organizer authorization and terms for redistribution of weights.
- [ ] Recover and qualify the exact runtime container/dependency environment.
- [x] Add source integrity, syntax and synthetic model smoke checks.
- [ ] Qualify full training and native-size inference in a fresh environment.
- [ ] Add a documented image example with redistribution rights.
- [ ] Coordinate repository organization and public-release timing with organizers.
- [ ] Prepare a MONAI bundle only after metadata, preprocessing and weights are verified.
- [ ] Confirm whether organizer plots may be placed in the public repository;
      permission for presentation use alone is not treated as repository permission.

Do not publish local proxy-trained checkpoints as the final challenge weights.
Do not include the organizer plots or private source data in the code release.
The author's private repository hosts this snapshot. Public release and any
change in visibility should follow organizer coordination.
