import paramiko
import os
import time
import re
import sys
import argparse
from contextlib import contextmanager

# --- Configuration ---
# The hostname for the HPC login node, which must match an entry in your ~/.ssh/config file.
HPC_HOSTNAME = 'hpc3.rcic.uci.edu'

# Base command to submit the Slurm job. The CPU count will be added dynamically.
SBATCH_SCRIPT_PATH = '/opt/rcic/scripts/vscode-sshd.sh'

# Time in seconds to wait between checking for the output file.
POLL_INTERVAL = 5

# Maximum time in seconds to wait for the output file before giving up.
MAX_WAIT_TIME = 300  # 5 minutes

# SSH connection timeout in seconds
SSH_TIMEOUT = 20

# Magic strings for job completion detection
JOB_COMPLETION_MARKER = "user mode sshd started"
OUTPUT_FILE_PREFIX = "vscode-sshd-"
OUTPUT_FILE_SUFFIX = ".out"

# Resource defaults
DEFAULT_GPU_TYPE = "A30"     # cluster default you’ve been using, there are also A100, L40S, V100
DEFAULT_GPU_COUNT = 1
DEFAULT_GPU_PARTITION = "free-gpu"
DEFAULT_FREE_PARTITION = "free"
DEFAULT_ACCOUNT = "pkaiser_lab"

# Retry configuration
MAX_RETRIES = 3                 # SSH command retries
RETRY_DELAY = 2                 # seconds between SSH retries
CREATE_MAX_ATTEMPTS = 3         # attempts to (re)submit on blacklist detection

# Node blacklist: these nodes will be excluded for GPU jobs by default
# (You can override via CLI: --blacklist hpc3-gpu-l54-08,hpc3-gpu-l54-XX)
BLACKLIST_NODES = ["hpc3-gpu-l54-08"]

# --------------------------------------------------------------------------------------
# Utility / Validation
# --------------------------------------------------------------------------------------

def validate_job_id(job_id):
    """
    Validates that a job ID is a positive integer.
    """
    if not job_id or not job_id.isdigit():
        return False
    return int(job_id) > 0

def sanitize_resource_param(param, param_name):
    """
    Sanitizes resource parameters to prevent command injection.
    """
    if not param:
        return None
    if any(char in param for char in [';', '&', '|', '$', '`', '(', ')', '<', '>', '"', "'"]):
        print(f"[!] Invalid characters in {param_name}: {param}", file=sys.stderr)
        return None
    return param.strip()

def sanitize_blacklist_nodes(nodes):
    """
    Sanitize node blacklist strings to safe characters for sbatch --exclude.
    Allows alnum, dash, underscore, dot, brackets, asterisk, comma.
    """
    safe = []
    for n in nodes:
        n = (n or "").strip()
        if not n:
            continue
        if re.fullmatch(r"[A-Za-z0-9._\-\[\]\*,]+", n):
            safe.append(n)
        else:
            print(f"[!] Skipping unsafe node name in blacklist: {n}", file=sys.stderr)
    # de-dup while preserving order
    seen = set()
    deduped = []
    for n in safe:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped

def execute_ssh_command_with_retry(client, command, max_retries=MAX_RETRIES):
    """
    Executes an SSH command with retry logic.
    """
    for attempt in range(max_retries + 1):
        try:
            stdin, stdout, stderr = client.exec_command(command)
            return stdin, stdout, stderr
        except Exception as e:
            if attempt < max_retries:
                print(f"[!] SSH command failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"    Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[!] SSH command failed after {max_retries + 1} attempts: {e}", file=sys.stderr)
                return None, None, None

@contextmanager
def ssh_client_context(config):
    """
    Context manager for SSH client that ensures proper cleanup.
    """
    client = None
    try:
        client = get_ssh_client(config)
        yield client
    except Exception as e:
        print(f"[!] SSH context error: {e}", file=sys.stderr)
        yield None
    finally:
        if client:
            try:
                print("[*] Closing SSH connection.")
                client.close()
            except Exception as e:
                print(f"[!] Error closing SSH connection: {e}", file=sys.stderr)

def get_ssh_client(config):
    """
    Creates and returns an authenticated SSH client using a parsed config.
    """
    hostname = config['hostname']
    username = config.get('user')

    if not username:
        print(f"[!] 'User' not specified in your SSH config for host '{HPC_HOSTNAME}'.", file=sys.stderr)
        return None

    try:
        print(f"[*] Connecting to {hostname} as {username} using your SSH config...")
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())

        client.connect(
            hostname=hostname,
            username=username,
            key_filename=config.get('identityfile'),
            sock=paramiko.ProxyCommand(config.get('proxycommand')) if config.get('proxycommand') else None,
            timeout=SSH_TIMEOUT
        )
        print("[+] Connection successful!")
        return client
    except Exception as e:
        print(f"[!] Connection failed: {e}", file=sys.stderr)
        print("[!] Please ensure your SSH key is added to your ssh-agent or is not passphrase protected.", file=sys.stderr)
        return None

# --------------------------------------------------------------------------------------
# Slurm helpers (submit, monitor, cancel)
# --------------------------------------------------------------------------------------

def submit_job(client, command):
    """
    Executes the sbatch command to submit the job and extracts the job ID.
    """
    try:
        print(f"[*] Submitting job with command: '{command}'")
        stdin, stdout, stderr = execute_ssh_command_with_retry(client, command)
        if not stdout:
            return None

        exit_status = stdout.channel.recv_exit_status()
        stdout_output = stdout.read().decode().strip()
        stderr_output = stderr.read().decode().strip()

        if exit_status != 0:
            print(f"[!] Error submitting job. Exit Status: {exit_status}", file=sys.stderr)
            if stderr_output:
                print(f"    Stderr: {stderr_output}", file=sys.stderr)
            return None

        print(f"[+] Server response: {stdout_output}")
        match = re.search(r'(\d+)', stdout_output)
        if match:
            job_id = match.group(1)
            print(f"[+] Successfully submitted job. Job ID: {job_id}")
            return job_id
        else:
            print("[!] Could not parse job ID from sbatch output.", file=sys.stderr)
            return None

    except Exception as e:
        print(f"[!] Failed to execute sbatch command: {e}", file=sys.stderr)
        return None

def cancel_job(client, job_id):
    """
    Executes the scancel command to cancel a running Slurm job.
    """
    command = f"scancel {job_id}"
    print(f"[*] Attempting to cancel job {job_id} with command: '{command}'")
    stdin, stdout, stderr = execute_ssh_command_with_retry(client, command)
    if not stdout:
        print(f"[!] Failed to execute cancel command for job {job_id}.", file=sys.stderr)
        return
    exit_status = stdout.channel.recv_exit_status()
    stderr_output = stderr.read().decode().strip()

    if exit_status == 0:
        print(f"[+] Successfully cancelled job {job_id}.")
    else:
        print(f"[!] Failed to cancel job {job_id}.", file=sys.stderr)
        if stderr_output:
            print(f"    Server error: {stderr_output}", file=sys.stderr)
        else:
            print("    Unknown error. The job may have already finished or the ID is invalid.", file=sys.stderr)

def squeue_job_state_and_nodes(client, job_id):
    """
    Returns (state, [nodes]) using squeue. Nodes may be empty if not assigned yet.
    """
    # %i jobid | %t state | %N nodelist
    cmd = f"squeue -j {job_id} -o '%i|%t|%N' -h"
    stdin, stdout, stderr = execute_ssh_command_with_retry(client, cmd)
    if not stdout:
        return None, []
    if stdout.channel.recv_exit_status() != 0:
        return None, []

    line = stdout.read().decode().strip()
    if not line:
        return None, []

    parts = line.split("|", 2)
    if len(parts) != 3:
        return None, []

    _jid, state, nodelist = parts
    nodelist = (nodelist or "").strip()

    # Nodelist may be "(none)" or blank while pending
    if not nodelist or nodelist.lower() in {"(none)", "(null)", "none", "n/a"}:
        return state, []

    # Could be "nodeA" or "nodeA,nodeB" or "node[01-03]" (but squeue expands typically)
    nodes = [n.strip() for n in re.split(r"[,\s]+", nodelist) if n.strip()]
    return state, nodes

def any_node_blacklisted(nodes, blacklist):
    """
    Returns True if any node is in the blacklist (exact match).
    """
    if not nodes or not blacklist:
        return False
    bl = set(blacklist)
    for n in nodes:
        if n in bl:
            return True
    return False

def get_job_output(client, job_id, blacklist_nodes=None, auto_cancel_on_blacklist=True):
    """
    Polls for the job output file, with a safety check to detect blacklisted node assignments.
    If a blacklisted node is detected, optionally cancels and returns the sentinel "__BLACKLISTED__".
    Otherwise, waits until JOB_COMPLETION_MARKER appears in the output file.
    """
    output_filename = f"{OUTPUT_FILE_PREFIX}{job_id}{OUTPUT_FILE_SUFFIX}"
    print(f"[*] Waiting for the complete output file '{output_filename}' to be created...")

    start_time = time.time()
    while time.time() - start_time < MAX_WAIT_TIME:
        # Check node assignment
        state, nodes = squeue_job_state_and_nodes(client, job_id)
        if nodes:
            print(f"    Job {job_id} state: {state}, node(s): {', '.join(nodes)}")
            if auto_cancel_on_blacklist and any_node_blacklisted(nodes, blacklist_nodes or []):
                print(f"[!] Job {job_id} landed on a blacklisted node ({', '.join(nodes)}). Cancelling and retrying...")
                cancel_job(client, job_id)
                return "__BLACKLISTED__"

        # Check output file contents
        check_command = f"ls {output_filename}"
        stdin, stdout, stderr = execute_ssh_command_with_retry(client, check_command)

        if stdout and stdout.channel.recv_exit_status() == 0:
            cat_command = f"cat {output_filename}"
            stdin, stdout, stderr = execute_ssh_command_with_retry(client, cat_command)

            if stdout.channel.recv_exit_status() == 0:
                content = stdout.read().decode()
                if JOB_COMPLETION_MARKER in content:
                    print("[+] Complete output file found!")
                    return content

        elapsed = int(time.time() - start_time)
        print(f"    ...still waiting for complete file (elapsed: {elapsed}s)")
        time.sleep(POLL_INTERVAL)

    print(f"[!] Timed out after {MAX_WAIT_TIME} seconds. Job may have failed to start or write output.", file=sys.stderr)
    return None

# --------------------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------------------

def parse_output_and_display(output_content, job_id, cpus, mem, gpu, pk_account, gpu_type, gpu_count):
    """
    Parses the job output to find the SSH config and prints it along with a cancel command.
    """
    if not output_content:
        print("[!] Cannot parse empty output content.", file=sys.stderr)
        return

    config_block_match = re.search(r"(Host hpc3-\*.*?StrictHostKeyChecking no)", output_content, re.DOTALL)

    if config_block_match:
        config_block = config_block_match.group(1).strip()
        print("\n" + "="*50)
        print("🎉 VS Code SSH Configuration Ready! 🎉")
        print("="*50)
        print("\nCopy the following block into your local SSH config file.")
        print("In VS Code, you can access this via 'Remote-SSH: Open Configuration File...'\n")
        print("-" * 50)
        print(config_block)
        print("-" * 50)
        print("\nAfter adding it, use 'Remote-SSH: Connect to Host...' and select 'hpc3-*'.")

        print("\n" + "="*50)

        # Build the job specification message dynamically
        job_spec_message = f"✅ Job {job_id} is running"
        requested_resources = []
        if gpu:
            requested_resources.append(f"{gpu_count} {gpu_type} GPU(s)")
        if cpus:
            requested_resources.append(f"{cpus} CPU(s)")
        if mem:
            requested_resources.append(f"{mem} of memory")

        if requested_resources:
            if len(requested_resources) > 2:
                resource_str = ", ".join(requested_resources[:-1]) + f", and {requested_resources[-1]}"
            else:
                resource_str = " and ".join(requested_resources)
            job_spec_message += f" with {resource_str}"

        if pk_account:
            job_spec_message += " under the 'pkaiser_lab' account"

        if not requested_resources and not pk_account:
            job_spec_message += " with default cluster resources"

        job_spec_message += "."

        print(job_spec_message)

        print(f"To stop this server later, run:")
        print(f"uv run python {os.path.basename(sys.argv[0])} cancel {job_id}")
        print("="*50)

    else:
        print("\n[!] Could not find the SSH configuration block in the job output.")
        print("    Please check the full job output below for errors:\n")
        print("-" * 50)
        print(output_content)
        print("-" * 50)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    """ Main function to parse arguments and run the automation script. """
    parser = argparse.ArgumentParser(
        description="A script to automate starting and stopping VS Code servers on the UCI HPC cluster via Slurm.",
        epilog="Example usage:\n"
               "  # Request a server with specific resources (CPU only)\n"
               "  uv run python hpc_automator.py create --cpus 8 --mem 32G\n\n"
               "  # Request a server with a GPU (CPU auto-assigned by cluster)\n"
               "  uv run python hpc_automator.py create --gpu\n\n"
               "  # Request a specific GPU type (CPU auto-assigned)\n"
               "  uv run python hpc_automator.py create --gpu --gpu-type A100\n\n"
               "  # Request a GPU with memory only (no CPU specification)\n"
               "  uv run python hpc_automator.py create --gpu --mem 16G\n\n"
               "  # Request a server in the 'free' partition\n"
               "  uv run python hpc_automator.py create --free\n\n"
               "  # Request a server under the 'pkaiser_lab' account\n"
               "  uv run python hpc_automator.py create --cpus 4 --pk_account\n\n"
               "  # Disable blacklist or customize it\n"
               "  uv run python hpc_automator.py create --gpu --no-blacklist\n"
               "  uv run python hpc_automator.py create --gpu --blacklist hpc3-gpu-l54-08,hpc3-gpu-l54-09\n\n"
               "  # Check current jobs\n"
               "  uv run python hpc_automator.py jobs\n\n"
               "  # Cancel a running server\n"
               "  uv run python hpc_automator.py cancel 123456",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available actions')

    # Create command
    parser_create = subparsers.add_parser(
        'create',
        help='Submits a new Slurm job to start a VS Code server.',
        description='Creates a new VS Code server job, waits for it to start, and provides the necessary SSH configuration.'
    )
    parser_create.add_argument('--cpus', type=int, help='Number of CPUs to request (e.g., 4). Cannot be used with --gpu.')
    parser_create.add_argument('--mem', type=str, help='Amount of memory to request (e.g., "16G").')
    parser_create.add_argument(
        '--gpu', action='store_true',
        help=f'Request a GPU for the job. Adds "-p {DEFAULT_GPU_PARTITION} --gres=gpu:{{GPU_TYPE}}:{DEFAULT_GPU_COUNT}". Cannot be used with --cpus.'
    )
    parser_create.add_argument(
        '--gpu-type', type=str, choices=['A30', 'A100', 'L40S', 'V100'], default=DEFAULT_GPU_TYPE,
        help=f'GPU type to request when using --gpu (default: {DEFAULT_GPU_TYPE}). Choices: A30, A100, L40S, V100.'
    )
    parser_create.add_argument(
        '--free', action='store_true',
        help=f'Request a job in the "{DEFAULT_FREE_PARTITION}" partition. Mutually exclusive with --gpu.'
    )
    parser_create.add_argument('--pk_account', action='store_true', help=f'Submit the job under the {DEFAULT_ACCOUNT} account.')
    parser_create.add_argument('--dry-run', action='store_true', help='Show the command without running it.')

    # Blacklist controls
    parser_create.add_argument(
        '--blacklist', type=str, default=None,
        help='Comma-separated node names to exclude (applied for --gpu by default).'
    )
    parser_create.add_argument(
        '--no-blacklist', action='store_true',
        help='Disable blacklist entirely (even for GPU jobs).'
    )
    parser_create.add_argument(
        '--apply-blacklist-to-all', action='store_true',
        help='Apply blacklist to CPU-only jobs as well (not just --gpu).'
    )

    # Cancel command
    parser_cancel = subparsers.add_parser(
        'cancel',
        help='Stops a running VS Code server job on Slurm.',
        description='Cancels a specific, running Slurm job using its job ID.'
    )
    parser_cancel.add_argument('job_id', help='The numeric ID of the Slurm job to be cancelled.')

    # Jobs command
    parser_jobs = subparsers.add_parser(
        'jobs',
        help='Checks current HPC jobs using squeue.',
        description='Displays current jobs for the user using squeue command.'
    )

    args = parser.parse_args()

    # --- Common setup for all actions ---
    print("--- UCI HPC VS Code Automation Script ---")

    # 1. Load SSH configuration from file
    ssh_config_path = os.path.expanduser('~/.ssh/config')
    try:
        with open(ssh_config_path) as f:
            ssh_config = paramiko.SSHConfig()
            ssh_config.parse(f)
    except FileNotFoundError:
        print(f"[!] SSH config file not found at '{ssh_config_path}'", file=sys.stderr)
        return

    user_config = ssh_config.lookup(HPC_HOSTNAME)
    if not user_config or 'hostname' not in user_config:
        print(f"[!] No configuration for host '{HPC_HOSTNAME}' found in '{ssh_config_path}'.", file=sys.stderr)
        return

    # Build blacklist list (only for commands that support these arguments)
    if hasattr(args, 'no_blacklist') and args.no_blacklist:
        active_blacklist = []
    else:
        if hasattr(args, 'blacklist') and args.blacklist:
            active_blacklist = sanitize_blacklist_nodes([x for x in args.blacklist.split(",")])
        else:
            active_blacklist = sanitize_blacklist_nodes(BLACKLIST_NODES)

    # 2. Handle dry-run mode without SSH connection
    if args.command == 'create' and getattr(args, 'dry_run', False):
        sbatch_options = []
        if args.pk_account:
            sbatch_options.append(f'--account={DEFAULT_ACCOUNT}')

        # partitions/GPUs
        if args.gpu:
            gpu_type = args.gpu_type if args.gpu_type else DEFAULT_GPU_TYPE
            sbatch_options.append(f'-p {DEFAULT_GPU_PARTITION} --gres=gpu:{gpu_type}:{DEFAULT_GPU_COUNT}')
            if active_blacklist:
                sbatch_options.append(f'--exclude={",".join(active_blacklist)}')
        elif args.free:
            sbatch_options.append(f'-p {DEFAULT_FREE_PARTITION}')
            if args.apply_blacklist_to_all and active_blacklist:
                sbatch_options.append(f'--exclude={",".join(active_blacklist)}')

        # resources - only add CPU for non-GPU jobs
        if args.cpus and not args.gpu:
            sbatch_options.append(f'--ntasks={args.cpus}')
        if args.mem:
            mem = sanitize_resource_param(args.mem, "memory")
            if not mem:
                return
            sbatch_options.append(f'--mem={mem}')

        sbatch_args_str = " ".join(sbatch_options)
        sbatch_command = f'sbatch {sbatch_args_str} {SBATCH_SCRIPT_PATH}'.strip()

        print("=== DRY RUN MODE ===")
        print(f"Would execute: {sbatch_command}")
        print("=== END DRY RUN ===")
        return

    # 3. Establish SSH connection with automatic cleanup
    with ssh_client_context(user_config) as client:
        if not client:
            return

        if args.command == 'create':
            # Validate mutual exclusivity of --gpu and --free
            if args.gpu and args.free:
                print("[!] Error: --gpu and --free options are mutually exclusive. Please choose one.", file=sys.stderr)
                return

            # Validate mutual exclusivity of --gpu and --cpus
            if args.gpu and args.cpus:
                print("[!] Error: --gpu and --cpus options are mutually exclusive. GPU jobs cannot specify CPU count.", file=sys.stderr)
                return

            # Validate and sanitize resource parameters
            if args.mem:
                args.mem = sanitize_resource_param(args.mem, "memory")
                if not args.mem:
                    return
            if args.cpus and args.cpus <= 0:
                print("[!] Error: CPU count must be a positive integer.", file=sys.stderr)
                return

            # Attempt loop (to recover if somehow scheduled to a blacklisted node)
            for attempt in range(1, CREATE_MAX_ATTEMPTS + 1):
                print(f"[*] Create attempt {attempt}/{CREATE_MAX_ATTEMPTS}")

                # Build sbatch options
                sbatch_options = []
                if args.pk_account:
                    sbatch_options.append(f'--account={DEFAULT_ACCOUNT}')

                if args.gpu:
                    gpu_type = args.gpu_type if args.gpu_type else DEFAULT_GPU_TYPE
                    sbatch_options.append(f'-p {DEFAULT_GPU_PARTITION} --gres=gpu:{gpu_type}:{DEFAULT_GPU_COUNT}')
                    if active_blacklist:
                        sbatch_options.append(f'--exclude={",".join(active_blacklist)}')
                elif args.free:
                    sbatch_options.append(f'-p {DEFAULT_FREE_PARTITION}')
                    if args.apply_blacklist_to_all and active_blacklist:
                        sbatch_options.append(f'--exclude={",".join(active_blacklist)}')
                else:
                    # no partition override; optionally apply blacklist if user asked for it
                    if args.apply_blacklist_to_all and active_blacklist:
                        sbatch_options.append(f'--exclude={",".join(active_blacklist)}')

                # Only add CPU request for non-GPU jobs to avoid scheduling conflicts
                if args.cpus and not args.gpu:
                    sbatch_options.append(f'--ntasks={args.cpus}')
                if args.mem:
                    sbatch_options.append(f'--mem={args.mem}')

                sbatch_args_str = " ".join(sbatch_options)
                sbatch_command = f'sbatch {sbatch_args_str} {SBATCH_SCRIPT_PATH}'.strip()

                job_id = submit_job(client, sbatch_command)
                if not job_id:
                    print("[!] Submission failed. Aborting.", file=sys.stderr)
                    return

                output_content = get_job_output(
                    client, job_id,
                    blacklist_nodes=active_blacklist,
                    auto_cancel_on_blacklist=True
                )

                if output_content == "__BLACKLISTED__":
                    # Go to next attempt (resubmit)
                    if attempt < CREATE_MAX_ATTEMPTS:
                        print("[*] Retrying create after blacklist cancellation...")
                        continue
                    else:
                        print("[!] Exceeded maximum attempts due to blacklist collisions.", file=sys.stderr)
                        return
                elif output_content:
                    gpu_type = args.gpu_type if args.gpu_type else DEFAULT_GPU_TYPE
                    parse_output_and_display(
                        output_content, job_id, args.cpus, args.mem, args.gpu,
                        args.pk_account, gpu_type, DEFAULT_GPU_COUNT
                    )
                    break
                else:
                    print("[!] Failed to retrieve job output. Please log in manually to check the job status.")
                    print(f"    Check for a file named '{OUTPUT_FILE_PREFIX}{job_id}{OUTPUT_FILE_SUFFIX}' in your home directory.")
                    break

        elif args.command == 'cancel':
            if not validate_job_id(args.job_id):
                print(f"[!] Invalid job ID: {args.job_id}. Must be a positive integer.", file=sys.stderr)
                return
            cancel_job(client, args.job_id)

        elif args.command == 'jobs':
            check_jobs(client)

def check_jobs(client, username="ddlin"):
    """
    Checks current jobs for the user using squeue command.
    """
    command = f"squeue -u {username}"
    print(f"[*] Checking current jobs with command: '{command}'")

    stdin, stdout, stderr = execute_ssh_command_with_retry(client, command)
    if not stdout:
        print("[!] Failed to execute squeue command.", file=sys.stderr)
        return

    exit_status = stdout.channel.recv_exit_status()
    stdout_output = stdout.read().decode().strip()
    stderr_output = stderr.read().decode().strip()

    if exit_status != 0:
        print(f"[!] Error checking jobs. Exit Status: {exit_status}", file=sys.stderr)
        if stderr_output:
            print(f"    Stderr: {stderr_output}", file=sys.stderr)
        return

    if not stdout_output:
        print("[+] No jobs currently running.")
        return

    print("\n" + "="*50)
    print("🔍 Current HPC Jobs")
    print("="*50)
    print(stdout_output)
    print("="*50)

if __name__ == '__main__':
    # Before running, ensure you have the 'paramiko' library installed.
    # pip install paramiko
    main()
