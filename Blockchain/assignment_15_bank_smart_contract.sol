pragma solidity ^0.4.19;

contract Bank {

    mapping(address => uint256) public balanceOf; // balances, indexed by addresses

    function deposit(uint256 amount) public payable {
       require(msg.value == amount);
       balanceOf[msg.sender] += amount;
    }

    function withdraw(uint256 amount) public {
      require(amount <= balanceOf[msg.sender]);
      balanceOf[msg.sender] -= amount;
      msg.sender.transfer(amount);
    }

    function getBalance() public view returns (uint256) {
      return balanceOf[msg.sender];
    }
}
