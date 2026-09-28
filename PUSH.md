# Putting this on GitHub

Copy this directory to your machine (e.g. `~/grad/cockpit-jax`), then:

```
bash push.sh
```

`push.sh` uses the GitHub CLI (`gh auth login` once, beforehand). Without `gh`, create
an empty public repo named `cockpit-jax` on github.com first, then:

```
cd ~/grad/cockpit-jax
git init -b main
git add .
git commit -m "cockpit-jax: unofficial partial JAX implementation of Cockpit, with checkpoint replay"
git remote add origin git@github.com:madyarc/cockpit-jax.git
git push -u origin main
```

## On crediting Claude

Two things that are often confused:

- **Collaborator** on GitHub means a real user account with write access. There is a
  `claude` account on github.com, and it belongs to some unrelated person; inviting it
  would invite a stranger to your repository. Anthropic publishes no account to add as
  a collaborator for this.
- **Credit** is what `push.sh` does: a `Co-authored-by: Claude <noreply@anthropic.com>`
  trailer on the commit, plus `AUTHORS.md` and the Authors section of the README.
  GitHub renders the trailer as a co-author on the commit. The address is a
  no-reply placeholder, so the co-author shows without a linked profile.

Keep the trailer on later commits with:

```
git commit -F - <<'MSG'
<subject line>

Co-authored-by: Claude <noreply@anthropic.com>
MSG
```
