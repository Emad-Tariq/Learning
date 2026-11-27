import unittest,tempfile,json,zipfile
from pathlib import Path
from persistence import save_bundle,restore_bundle,REQUIRED,digest


class PersistenceTests(unittest.TestCase):
    def setup_run(self,root):
        run=root/'run';code=root/'code';run.mkdir();code.mkdir()
        for name in REQUIRED:(run/name).write_bytes(('saved '+name).encode())
        (run/'config.json').write_text(json.dumps({'primary':'mlp_blend'}))
        (run/'sift_mutual_example.npz').write_bytes(b'checked pairs including zeros')
        (code/'verifier.py').write_text('# frozen verifier')
        return run,code

    def test_roundtrip_models_sources_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,code=self.setup_run(root);archive=root/'bundle.zip'
            m=save_bundle(run,code,archive);self.assertTrue(m['complete_for_submission'])
            restore_bundle(archive,root/'restored',root/'restored_code')
            for file in run.iterdir():self.assertEqual(digest(file),digest(root/'restored'/file.name))
            self.assertEqual(digest(code/'verifier.py'),digest(root/'restored_code/verifier.py'))

    def test_partial_backup_explicitly_reports_missing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,code=self.setup_run(root);(run/'calibrators.joblib').unlink()
            m=save_bundle(run,code,root/'bundle.zip')
            self.assertFalse(m['complete_for_submission']);self.assertIn('calibrators.joblib',m['missing_for_submission'])

    def test_corruption_rejected_before_any_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,code=self.setup_run(root);archive=root/'bundle.zip'
            save_bundle(run,code,archive)
            with zipfile.ZipFile(archive) as z:members={name:z.read(name) for name in z.namelist()}
            members['run/verifier.joblib']=b'corrupted'
            with zipfile.ZipFile(root/'corrupt.zip','w') as z:
                for name,raw in members.items():z.writestr(name,raw)
            with self.assertRaises(ValueError):restore_bundle(root/'corrupt.zip',root/'restored',root/'restored_code')
            self.assertFalse((root/'restored').exists())

    def test_existing_different_checkpoint_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,code=self.setup_run(root);archive=root/'bundle.zip'
            save_bundle(run,code,archive);dest=root/'existing';dest.mkdir()
            (dest/'verifier.joblib').write_bytes(b'another run')
            with self.assertRaises(ValueError):restore_bundle(archive,dest,root/'restored_code')
            self.assertEqual((dest/'verifier.joblib').read_bytes(),b'another run')


if __name__=='__main__':unittest.main()
