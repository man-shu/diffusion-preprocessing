import unittest
import m_niflow_multiproc_lite
import os 

class TestNiflow(unittest.TestCase):

    def setUp(self):
        self.flip = (-1, 1, 1)
        self.bvecs_in = ('/data/parietal/store/work/zmohamed/mathfun/raw_data_visit1/dti_raw.bvecs')


    def test_bvec_flip_valid(self):

        file_path = m_niflow_multiproc_lite.bvec_flip(self.bvecs_in, self.flip)

        # Check if the file path exists
        self.assertIsInstance(file_path, str)
        self.assertTrue(os.path.exists(file_path))

  

if __name__ == '__main__':
    unittest.main()