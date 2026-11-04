import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
import numpy as np
from WaterPS import load_potential

class PotentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=json.loads(Path(__file__).with_name('reference.json').read_text())
        cls.coords=np.array(data['coords']);cls.reference=np.array(data['energy'])
        cls.pes=load_potential(os.environ.get('POTENTIAL_LIBRARY'))

    def test_00_concurrent_first_calls(self):
        barrier=threading.Barrier(8)
        def call(i):
            barrier.wait(timeout=30)
            return i,self.pes.get_pot(coords=self.coords[i])
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,result in pool.map(call,range(8)):
                self.assertAlmostEqual(result,self.reference[i],delta=1e-13)

    def test_serial_reference_and_batch(self):
        np.testing.assert_allclose(self.pes.get_pot(coords=self.coords),self.reference,rtol=0,atol=1e-13)
        result=self.pes.get_pot(coords=self.coords.reshape(2,8,3,3))
        np.testing.assert_allclose(result,self.reference.reshape(2,8),rtol=0,atol=1e-13)
        np.testing.assert_allclose(self.pes.get_pot(coords=self.coords[::-1]),self.reference[::-1],rtol=0,atol=1e-13)
        for i,x in enumerate(self.coords):
            self.assertAlmostEqual(self.pes.get_pot(coords=x),self.reference[i],delta=1e-13)

    def test_thread_stress(self):
        expected=self.pes.get_pot(coords=self.coords)
        def call(i):
            index=i%len(self.coords)
            self.assertEqual(self.pes.get_pot(coords=self.coords[index]),expected[index])
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(call,range(2048)))

    def test_input_and_ownership(self):
        original=self.coords.copy()
        a=self.pes.get_pot(coords=self.coords);b=self.pes.get_pot(coords=self.coords)
        self.assertFalse(np.shares_memory(a,b))
        np.testing.assert_array_equal(original,self.coords)
        self.assertEqual(self.pes.get_pot(coords=np.empty((0,3,3))).shape,(0,))
        for x in [np.zeros((3,3)),np.zeros(9),np.full((3,3),np.inf)]:
            with self.assertRaises(ValueError): self.pes.get_pot(coords=x)

if __name__=='__main__': unittest.main()
