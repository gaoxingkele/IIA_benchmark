"""Keep released-data and paper-Sines full artifact reviews explicitly separate.

Only cohort admission changes from v1. The existing guarded controller and the
unchanged full tensor/array/four-evaluator audit still execute in native Py310.
This is an artifact audit, not another generation or learned-evaluator run.
"""
from scripts.flow_matching import audit_spectral_completed_cohorts_v1 as original


def validate_selection(config, queue, audits):
    track = config['completed_data_track']
    if track not in {'released_author_data', 'paper_sines_distribution'}:
        raise ValueError('An explicit registered data track is required')
    jobs = {job['id']: job for job in queue['jobs']}
    if (len(queue['jobs']) != 70 or len(jobs) != 70
            or config['original_required_seeds'] != [42, 43, 44, 45, 46]):
        raise ValueError('Preserve all70 unique jobs and original five-seed obligations')
    cohorts = config['completed_cohort_seeds']
    if (not cohorts or any(not seeds or len(seeds) != len(set(seeds))
            or not set(seeds).issubset(config['original_required_seeds'])
            for seeds in cohorts.values())):
        raise ValueError('Every requested original seed must remain explicit')
    if track == 'paper_sines_distribution' and set(cohorts) != {'sines'}:
        raise ValueError('The paper-distribution track is registered only for Sines')
    required = {job['id'] for job in jobs.values()
        if job['model'] == 'diffusion_ts' and job['data_track'] == track
        and job['dataset'] in cohorts and job['seed'] in cohorts[job['dataset']]}
    if len(required) != sum(len(seeds) for seeds in cohorts.values()):
        raise ValueError('Every requested complete cohort member must be registered')
    selected = [entry['id'] for entry in config['audits']]
    if not required or len(selected) != len(set(selected)) or set(selected) != required:
        raise ValueError('All selected complete models must match the exact data track')
    for entry in config['audits']:
        job, audit = jobs[entry['id']], audits[entry['id']]
        if (audit['queue'] != config['original_queue'] or audit['job_id'] != job['id']
                or audit['terminal_receipt'] != queue['state_root'] + '/receipts/' + job['id'] + '.json'
                or job['generation_batch_size'] != 2001 or job['generation_field_chunk_size'] is not None):
            raise ValueError('Full original identity, sampling recipe and source guard required')
    for key in ['minimum_available_memory_bytes', 'minimum_commit_headroom_bytes',
                'emergency_commit_headroom_bytes']:
        if config['settings'][key] != queue['settings'][key]:
            raise ValueError('Preserve original full native memory safeguards')


def main():
    # This process alone uses the new admission check. No existing source file,
    # live controller, numerical worker or artifact-audit function is modified.
    original.validate_selection = validate_selection
    original.main()


if __name__ == '__main__':
    main()
