# Putting Lumina Advisors online (a permanent, real URL)

This gets you a real link like `https://lumina-advisors.onrender.com` that
works from any device, anytime, without your laptop needing to be on. It
uses **Render.com**, because its free tier needs no credit card and it
understands plain Python projects like this one with almost no setup.

Everything in the repo is already prepared (`render.yaml`, `.gitignore`,
`.env.example`). The steps below are the ones only you can do, because they
create accounts under your name -- I can't create accounts on your behalf.
It's about 15 minutes, once.

## 1. Put the code on GitHub

Render deploys from a GitHub repo, so the code needs to live there first.

1. Go to **github.com** and sign up if you don't have an account (it's
   free).
2. Click the **+** in the top right -> **New repository**. Name it
   `lumina-advisors`, keep it **Public** or **Private** (either works),
   and don't check any of the "initialize with" boxes (README, .gitignore,
   license) -- this project already has its own.
3. Click **Create repository**. GitHub will show you a page with commands
   under "...or push an existing repository from the command line" --
   you won't need most of that; the important part is the URL, which
   looks like `https://github.com/YOUR-USERNAME/lumina-advisors.git`.
4. On your computer, open the `stockgraph` folder in a terminal (or ask me
   to do this part with you -- pushing to a repo you just created isn't
   one of the things I need to hold back on) and run:
   ```
   git remote add origin https://github.com/YOUR-USERNAME/lumina-advisors.git
   git branch -M main
   git push -u origin main
   ```
   GitHub will ask you to sign in (a browser popup or a personal access
   token) -- follow its prompts.

## 2. Create a Render account and connect the repo

1. Go to **render.com** -> **Get Started** -> sign up (you can use your
   GitHub account to sign up, which also handles step 3 below in one
   click). No credit card needed for the free plan.
2. Once logged in, click **New +** -> **Blueprint**.
3. Connect your GitHub account if prompted, then pick the
   `lumina-advisors` repo you just created.
4. Render will read `render.yaml` from the repo automatically and show you
   a preview of one web service called `lumina-advisors`. Click
   **Apply** / **Create**.
5. First deploy takes a few minutes (installing pandas/numpy, then running
   the startup pipeline). When it says **Live**, your URL is shown at the
   top of the service page (something like
   `https://lumina-advisors.onrender.com`) -- that's the permanent link.

## 3. (Optional) Turn on the AI narrative + chat widget

The site works fully without this -- forecasts, the graph, the scorecard,
accounts/watchlists all work either way. If you want Claude's plain-English
narrative and the chat widget to work on the live site too:

1. In the Render dashboard, open the `lumina-advisors` service ->
   **Environment** tab.
2. Add an environment variable `ANTHROPIC_API_KEY` with a key from
   **console.anthropic.com** (this needs its own account + billing there;
   it's a separate product from this site).
3. Save -- Render redeploys automatically.

## What to know about the free plan

- **No card required** -- this is genuinely free, not a trial.
- **It sleeps.** After ~15 minutes with no visitors, Render spins the site
  down. The next visit wakes it back up, which takes roughly 30-60 seconds
  (that's the site's own startup pipeline running -- pulling data and
  training the forecast models -- before it can answer). After that it's
  fast again until it goes quiet for another 15 minutes. This is normal
  and not a bug.
- **Accounts/watchlists reset on redeploy.** The database is a single
  SQLite file living on the server's local disk, which Render's free plan
  does not keep across deploys (pushing new code, or Render's own periodic
  maintenance, wipes it). Signing in and starring companies works great
  day-to-day; just know that pushing an update to the code will reset
  whoever had signed up. If this ever matters enough to fix, the real fix
  is a Render **paid** plan with a persistent disk, or moving to a small
  hosted Postgres database instead of SQLite -- not something to worry
  about for a portfolio/demo site.
- **Data mode is `auto`** (same default as running it locally): it tries
  live Yahoo/SEC/Google News data per company and quietly fills in
  synthetic data only for whatever it can't reach, so the site still
  looks complete even if Render's shared IP gets rate-limited by one of
  those sources sometimes. The badge in the top-right always tells you
  which is actually active.

## Updating the live site later

Any time you (or I) push new commits to the `main` branch on GitHub,
Render redeploys automatically within a minute or two -- no extra steps.

## If you'd rather not use Render

Everything above is Render-specific, but the app itself has no Render
dependency -- `render.yaml` is just a convenience file Render reads.
Railway and Fly.io both work too if you'd rather use them, though as of
this writing neither has as generous a truly-free tier as Render's (they
tend to ask for a card, even if usage stays inside a free allowance). The
one thing to keep straight if you switch: `python run.py` is the start
command, and it reads the assigned port from a `PORT` environment
variable, which Render/Railway/Fly.io all set automatically.

**Want your own server instead** -- a real domain, no cold-start sleep, and
a database that survives redeploys, at the cost of more setup and usually a
few dollars a month? See `deploy/DEPLOY.md` for a fully prepared systemd +
Caddy (free auto-HTTPS) setup for a plain Ubuntu VPS (Oracle's free tier,
DigitalOcean, or Hetzner all work) -- just SSH in and run
`deploy/setup.sh`.
