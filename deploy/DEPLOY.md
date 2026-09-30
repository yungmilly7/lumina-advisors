# Deploying Lumina Advisors on your own server (advanced option)

> **Looking for the easy path?** See `/DEPLOY.md` in the project root first
> — it walks through Render.com, which is free, needs no card, and no SSH.
> Come back to this file only if you want your own server instead: more
> control (a real domain, no cold-start sleep, a database that survives
> redeploys) at the cost of more setup and, usually, a few dollars a month.

Right now the site only runs on your own machine (`localhost`). This is the
path to a real URL other people can open. Everything here is prepared and
tested to be copy-paste; the only steps that have to be *you* are signing up
for a server and (optionally) a domain, since those need your own payment
details and identity — not something I can do on your behalf.

## 1. Get a server (5-10 minutes, this part is on you)

Pick one:

- **Free**: [Oracle Cloud Always Free tier](https://www.oracle.com/cloud/free/)
  — a genuinely free-forever ARM VM (2 OCPU / 12 GB RAM as of 2026, plenty
  for this app). Sign-up asks for a card for identity verification but
  won't charge you unless you explicitly upgrade. Create an "Ampere A1"
  instance running Ubuntu 22.04 or 24.04.
- **Cheap and simple** (~$6/mo): [DigitalOcean](https://www.digitalocean.com/)
  Basic Droplet, 1 GB RAM, Ubuntu 24.04. Fewer setup quirks than Oracle's
  free tier if you'd rather not fight ARM/networking edge cases.
- **Cheapest paid** (~$4-7/mo): [Hetzner](https://www.hetzner.com/cloud/) CX23
  or the ARM CAX11 — best value if you don't mind a non-US data center.

Whichever you pick, when it's created you'll get a **public IP address** and
either a password or an SSH key to log in with.

## 2. Point a domain at it (optional, but recommended before sharing it widely)

If you don't already have a domain, [Namecheap](https://www.namecheap.com/)
or [Cloudflare](https://www.cloudflare.com/products/registrar/) are the
usual cheap options (~$10-15/year for a `.com`). Once you have one, add an
**A record** pointing at your server's IP. Skip this and use the bare-IP
Caddy config (see `deploy/Caddyfile`) if you just want to test with a few
people first — no domain required for that.

## 3. Get the code onto the server and run the setup script

SSH into your new server (`ssh root@<your-server-ip>`), then either `git
clone` this project or upload it, so it ends up at `/opt/stockgraph`. Then:

```bash
cd /opt/stockgraph
sudo bash deploy/setup.sh
```

This installs Python, numpy/pandas, and Caddy; creates an unprivileged
`stockgraph` user to run the app as (not root); and installs both as
systemd services so they survive reboots and restart automatically if they
crash. It'll tell you exactly what's left to do when it finishes.

## 4. Fill in `.env` and start it

```bash
sudo nano /opt/stockgraph/.env
```

Set `SEC_USER_AGENT`, `ANTHROPIC_API_KEY` (if you want the AI narrative/chat
on), and `STOCKGRAPH_DATA_MODE=auto`. Then edit `/etc/caddy/Caddyfile` with
your real domain (or use the bare-IP block if you skipped step 2). Finally:

```bash
sudo systemctl start stockgraph
sudo systemctl start caddy
sudo journalctl -u stockgraph -f   # watch it boot -- first run takes a
                                    # few minutes while it ingests + trains
```

Once you see `bootstrap complete` in the logs, your site is live at your
domain (or `http://<server-ip>` if you skipped the domain step).

## What I already handled

- `deploy/stockgraph.service` — runs the app as a non-root user, restarts it
  automatically if it ever crashes, and only exposes it to `localhost` (not
  directly to the internet).
- `deploy/Caddyfile` — free automatic HTTPS via Let's Encrypt, plus a few
  standard security headers. No certificate wrangling needed.
- `deploy/setup.sh` — does every install/config step above in one command.

## What's still worth doing before a real public launch

- **Backups**: the SQLite database (`data/stockgraph.db`) holds every user's
  account + watchlist. A simple cron job copying it to another location (or
  your host's snapshot feature) is cheap insurance.
- **A second, non-root SSH login** with a real SSH key (not just root/
  password) — most providers set this up for you during signup.
- **Rate limiting at the network level** eventually, if it gets real
  traffic — the app has its own login/signup limiters, but a reverse-proxy-
  level limit (Caddy or Cloudflare) is a good second layer.

Tell me once you've got a server up and I'll walk through the rest with you,
or take it from there directly if you give me SSH access.
