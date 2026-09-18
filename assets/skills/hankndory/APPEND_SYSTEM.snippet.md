# >>> brainrouter-managed (hankndory) v1 >>>
### Brainrouter review = HankNDory Step 9 (post-implementation)
- Brainrouter's `request_review` is the post-implementation "mean code review" (HankNDory Step 9). It does NOT replace the pre-implementation Dory gates (comprehension, critic, readiness), which must still run in separate, fresh sessions.
- Keeping review-feedback iteration in one session applies to this post-implementation review loop ONLY. Dory validation phases must each run in their own fresh session.
- The reviewer may run on cloud OR local; Brainrouter chooses the backend. (Do not assume "local".)
- If `request_review` returns `status: "disabled"`, the code reviewer is turned OFF in Brainrouter; committing without a review is allowed in that mode.
# <<< brainrouter-managed <<<
