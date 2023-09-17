import unittest
import m_niflow_multiproc_lite
import os 

class TestNiflow(unittest.TestCase):


    def test_bvec_flip_valid(self):

        flip = (-1, 1, 1)
        bvecs_in= ('/data/parietal/store/work/zmohamed/mathfun/raw_data_visit1/dti_raw.bvecs')

        file_path = m_niflow_multiproc_lite.bvec_flip(bvecs_in, flip)

        # Check if the file path exists
        self.assertIsInstance(file_path, str)
        self.assertTrue(os.path.exists(file_path))

  

if __name__ == '__main__':
    unittest.main()