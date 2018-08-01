updated 20180719

---

data folder structure should be
 * data_root
   * train
     * image
     * mask
   * test
     * image
     * mask

file names of one image and its mask should be identical

---
dataloader should have 3 inputs
1. root_path : root to your data folder, it should contain 2 folder train and test, it can be a list of root.
2. transforms : compose transform you need.
3. train : boolean, indicate which set i want to load, true for train folder ; false for test folder.

