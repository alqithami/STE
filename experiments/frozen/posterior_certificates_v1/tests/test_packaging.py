"""Packaging must preserve the already verified audit boundary."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from package_results import digest, make_archive, validate_audited_results


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)/'production'
        self.out.mkdir()
        (self.out/'prediction.txt').write_bytes(b'audited prediction\n')
        fit = {'status': 'FIT_COMPLETE_AWAITING_AUDIT',
               'hashes': {'prediction.txt': digest((self.out/'prediction.txt').read_bytes())}}
        self.write_json('FIT_COMPLETE.json', fit)
        report = {'status': 'AUDIT_COMPLETE', 'coverage_verified': True,
                  'source_hashes_verified': True, 'data_hashes_verified': True,
                  'checkpoint_hashes_verified': True, 'selected_replay_verified': True,
                  'classification': 'exploratory_descriptive_benchmark',
                  'EQUAL_ALLOCATION_VALID': True, 'mode': 'production'}
        self.write_json('AUDIT.json', report)
        self.write_json('COMPLETE.json', {**report, 'status': 'COMPLETE',
            'FIT_COMPLETE_sha256': digest((self.out/'FIT_COMPLETE.json').read_bytes()),
            'AUDIT_sha256': digest((self.out/'AUDIT.json').read_bytes())})

    def write_json(self, name, value):
        (self.out/name).write_text(json.dumps(value, sort_keys=True))

    def test_valid_audit_chain_and_payload(self):
        self.assertEqual(validate_audited_results(self.out)['status'], 'COMPLETE')

    def test_tampered_audit_marker_is_rejected(self):
        report = json.loads((self.out/'AUDIT.json').read_text())
        report['classification'] = 'tampered'
        self.write_json('AUDIT.json', report)
        with self.assertRaisesRegex(RuntimeError, 'Changed audit-chain marker'):
            validate_audited_results(self.out)

    def test_tampered_complete_report_is_rejected(self):
        report = json.loads((self.out/'COMPLETE.json').read_text())
        report['classification'] = 'tampered'
        self.write_json('COMPLETE.json', report)
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE disagrees with AUDIT'):
            validate_audited_results(self.out)

    def test_unverified_marker_is_rejected_even_with_updated_hash(self):
        report = json.loads((self.out/'AUDIT.json').read_text())
        report['selected_replay_verified'] = False
        self.write_json('AUDIT.json', report)
        complete = json.loads((self.out/'COMPLETE.json').read_text())
        complete['selected_replay_verified'] = False
        complete['AUDIT_sha256'] = digest((self.out/'AUDIT.json').read_bytes())
        self.write_json('COMPLETE.json', complete)
        with self.assertRaisesRegex(RuntimeError, 'Required audit verification missing'):
            validate_audited_results(self.out)

    def test_unhashed_file_is_rejected(self):
        (self.out/'unexpected.txt').write_text('not covered by the audit')
        with self.assertRaisesRegex(RuntimeError, 'Unhashed or missing'):
            validate_audited_results(self.out)

    def test_missing_payload_is_rejected(self):
        (self.out/'prediction.txt').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Unhashed or missing'):
            validate_audited_results(self.out)

    def test_payload_hash_mismatch_is_rejected(self):
        (self.out/'prediction.txt').write_bytes(b'changed prediction\n')
        with self.assertRaisesRegex(RuntimeError, 'Changed audited payload'):
            validate_audited_results(self.out)

    def test_archive_manifest_matches_payload_and_external_digest(self):
        path = Path(self.temp.name)/'results.zip'
        payload = {'results/prediction.txt': b'audited prediction\n'}
        record = make_archive(path, payload, {'classification': 'software_QA'})
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read('ARCHIVE_MANIFEST.json'))
            self.assertEqual(archive.read('results/prediction.txt'), payload['results/prediction.txt'])
            self.assertEqual(manifest['payload_sha256'],
                             {name: digest(value) for name, value in payload.items()})
            self.assertIsNone(archive.testzip())
        self.assertEqual(record['sha256'], digest(path.read_bytes()))
        self.assertEqual(path.with_suffix('.zip.sha256').read_text(),
                         record['sha256']+'  results.zip\n')


if __name__ == '__main__':
    unittest.main()
