pragma solidity >=0.5.0 <0.6.0;

contract graph {
    address owner = msg.sender;
    // uint256 g = 65537;
    uint256 acc = 0;
    uint256 n = 0;

    modifier onlyOwner{
        require (msg.sender == owner);
        _;
    }
    
    function setGnN(uint x, uint y) public onlyOwner{
        if (acc == 0)
            acc = x;
        if (n == 0)
            n = y;
    }

    function add(uint x) public onlyOwner{
        acc = mexp(acc, x, n);
    }
    
    function batchAdd(uint[] memory xList) public onlyOwner{
        for (uint i = 0; i < xList.length; i++) {
            acc = mexp(acc, xList[i], n);
        }
    }
    
    function del(uint xInv) public onlyOwner{
        acc = mexp(acc, xInv, n);
    }
    
    function batchDel(uint[] memory xInvList) public onlyOwner{
        for (uint i = 0; i < xInvList.length; i++) {
            acc = mexp(acc, xInvList[i], n);
        }
    }
    
    function update(uint[] memory xInvList, uint[] memory xAddList) public onlyOwner{
        batchDel(xInvList);
        batchAdd(xAddList);
    }
    
    function getAcc() public view returns (uint256){
        return acc;
    }
    
    /**
     * @dev Compute modular exponential (x ** k) % m
     * @param x k m
     * @return uint
     */
   function mexp(uint x, uint k, uint m) internal pure returns (uint r) {
       r = 1;
       for (uint s = 1; s <= k; s *= 2) {
            if (k & s != 0) r = mulmod(r, x, m);
            x = mulmod(x, x, m);
       }
    }
}