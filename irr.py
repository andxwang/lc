"""
Revenue-based loan management system with IRR targeting.

This module implements a portfolio management system for revenue-based loans
where borrowers repay a target amount by sharing a percentage of their transaction
revenue with the lender. The system determines the required revenue share rate
to achieve a target annualized IRR.
"""

from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.optimize import brentq
import math


@dataclass
class Loan:
    """
    Represents a single revenue-based loan.
    
    Attributes:
        loan_id: Unique identifier for the loan.
        principal: Amount advanced at origination (lender outflow).
        fee: Fixed fee on top of principal (target_repayment = principal + fee).
        duration_days: Number of days before post-maturity sweep is triggered.
        target_annual_irr: Target annualized IRR for the loan.
        irr_function: Callable that computes annualized IRR from daily cash flows.
        transactions: Dict mapping day to list of (transaction_id, gross_amount) tuples.
        loan_status: One of 'active', 'repaid', 'post_maturity'.
        revenue_share_rate: The determined rate such that IRR ≈ target_annual_irr.
    """
    
    loan_id: str
    principal: float
    fee: float
    duration_days: int
    target_annual_irr: float
    irr_function: Callable[[List[float]], float]
    
    transactions: Dict[int, List[Tuple[str, float]]] = field(default_factory=lambda: defaultdict(list))
    loan_status: str = 'active'  # 'active', 'repaid', 'post_maturity'
    revenue_share_rate: Optional[float] = None
    current_day: int = 0
    
    @property
    def target_repayment(self) -> float:
        """Total amount borrower must repay: principal + fee."""
        return self.principal + self.fee
    
    def add_transaction(self, day: int, txn_id: str, gross_amount: float) -> None:
        """
        Record a transaction for a specific day.
        
        Args:
            day: Day number on which transaction occurs.
            txn_id: Unique transaction identifier.
            gross_amount: Gross transaction amount before revenue share.
        """
        self.transactions[day].append((txn_id, gross_amount))
        self.current_day = max(self.current_day, day)
    
    def simulate_cash_flows(
        self,
        revenue_share_rate: float,
        transactions: Optional[Dict[int, List[Tuple[str, float]]]] = None
    ) -> List[float]:
        """
        Simulate the cash flow sequence for a given revenue share rate.
        
        Args:
            revenue_share_rate: The rate at which borrower shares revenue (0 to 1).
            transactions: Optional override of transactions dict for simulation.
        
        Returns:
            List of daily cash flows where index is day number.
            Day 0 is the principal outflow (negative).
            Positive values are inflows to lender.
        """
        txn_dict = transactions if transactions is not None else self.transactions
        
        # Determine the last day we need to simulate
        if not txn_dict:
            last_txn_day = 0
        else:
            last_txn_day = max(txn_dict.keys())
        
        # Initialize cash flows: Day 0 is principal outflow
        max_day = max(last_txn_day, self.duration_days)
        cash_flows = [0.0] * (max_day + 1)
        cash_flows[0] = -self.principal
        
        outstanding_balance = self.target_repayment
        
        # Process each day
        for day in range(1, max_day + 1):
            if outstanding_balance <= 0:
                break
            
            # Sum transactions for this day
            daily_gross = sum(amount for _, amount in txn_dict.get(day, []))
            
            # Determine applicable revenue share rate
            if day <= self.duration_days:
                # Pre-maturity: use candidate rate
                applicable_rate = revenue_share_rate
            else:
                # Post-maturity: 100% revenue capture until repaid
                applicable_rate = 1.0
            
            # Calculate payment to lender
            payment_to_lender = applicable_rate * daily_gross
            
            # Cap payment at outstanding balance
            payment_to_lender = min(payment_to_lender, outstanding_balance)
            
            # Record cash flow
            cash_flows[day] = payment_to_lender
            outstanding_balance -= payment_to_lender
        
        # Trim trailing zeros for efficiency
        while len(cash_flows) > 1 and cash_flows[-1] == 0:
            cash_flows.pop()
        
        return cash_flows
    
    def compute_irr(self, revenue_share_rate: float) -> float:
        """
        Compute annualized IRR for a given revenue share rate.
        
        Args:
            revenue_share_rate: The rate at which borrower shares revenue.
        
        Returns:
            Annualized IRR.
        """
        cash_flows = self.simulate_cash_flows(revenue_share_rate)
        return self.irr_function(cash_flows)
    
    def objective_function(self, revenue_share_rate: float) -> float:
        """
        Objective function for root-finding: IRR(rate) - target_IRR.
        
        Args:
            revenue_share_rate: The rate to evaluate.
        
        Returns:
            Difference between achieved IRR and target IRR.
        """
        achieved_irr = self.compute_irr(revenue_share_rate)
        return achieved_irr - self.target_annual_irr
    
    def determine_revenue_share_rate(self) -> float:
        """
        Solve for the revenue share rate that achieves the target IRR.
        
        Uses Brent's method to find the root of the objective function.
        The revenue share rate must be in [0, 1].
        
        Returns:
            The determined revenue share rate.
        
        Raises:
            ValueError: If the target IRR is infeasible (cannot be achieved
                       even with rate = 1.0).
        """
        # Check feasibility: can we achieve target IRR with rate = 1.0?
        irr_at_max = self.objective_function(1.0)
        irr_at_min = self.objective_function(0.0)
        
        # If both have the same sign, the target might be infeasible
        if irr_at_max < 0:
            raise ValueError(
                f"Loan {self.loan_id}: Target IRR {self.target_annual_irr} is infeasible. "
                f"Even at 100% revenue share, IRR = {self.compute_irr(1.0)}"
            )
        
        if irr_at_min > 0:
            # Target is achievable at very low rates; clamp to near-zero
            self.revenue_share_rate = 1e-6
            return self.revenue_share_rate
        
        # Use Brent's method to find the root
        try:
            rate = brentq(self.objective_function, 0.0, 1.0, xtol=1e-6)
            self.revenue_share_rate = rate
            return rate
        except ValueError as e:
            raise ValueError(
                f"Loan {self.loan_id}: Failed to determine revenue share rate. {e}"
            )
    
    def process_txn(
        self,
        day: int,
        txn_id: str,
        gross_amount: float
    ) -> Dict[str, float]:
        """
        Process a single transaction, using the pre-determined revenue share rate.
        
        Args:
            day: Day on which transaction occurs.
            txn_id: Unique transaction identifier.
            gross_amount: Gross transaction amount.
        
        Returns:
            Dict with keys:
                'txn_id': Transaction ID.
                'gross_amount': Gross amount.
                'lender_share': Amount paid to lender.
                'borrower_keeps': Amount kept by borrower.
                'effective_rate': Revenue share rate applied.
        
        Raises:
            RuntimeError: If revenue_share_rate has not been determined yet.
        """
        if self.revenue_share_rate is None:
            raise RuntimeError(
                f"Loan {self.loan_id}: Revenue share rate not yet determined. "
                "Call determine_revenue_share_rate() first."
            )
        
        # Record the transaction
        self.add_transaction(day, txn_id, gross_amount)
        
        # Determine applicable rate
        if day <= self.duration_days:
            applicable_rate = self.revenue_share_rate
            status = 'active'
        else:
            applicable_rate = 1.0
            status = 'post_maturity'
        
        # Calculate amounts
        lender_share = applicable_rate * gross_amount
        borrower_keeps = (1 - applicable_rate) * gross_amount
        
        return {
            'txn_id': txn_id,
            'gross_amount': gross_amount,
            'lender_share': lender_share,
            'borrower_keeps': borrower_keeps,
            'effective_rate': applicable_rate,
            'loan_status': status
        }


class LoanManager:
    """
    Manages a portfolio of multiple revenue-based loans.
    
    Routes transactions to appropriate loans, manages loan lifecycle,
    and can compute portfolio-level metrics.
    """
    
    def __init__(self):
        """Initialize the loan manager with an empty portfolio."""
        self.loans: Dict[str, Loan] = {}
    
    def create_loan(
        self,
        loan_id: str,
        principal: float,
        fee: float,
        duration_days: int,
        target_annual_irr: float,
        irr_function: Callable[[List[float]], float]
    ) -> Loan:
        """
        Create a new loan in the portfolio.
        
        Args:
            loan_id: Unique identifier for the loan.
            principal: Principal amount advanced.
            fee: Fixed fee (total repayment = principal + fee).
            duration_days: Loan duration in days before post-maturity sweep.
            target_annual_irr: Target annualized IRR.
            irr_function: Function to compute IRR from cash flows.
        
        Returns:
            The created Loan object.
        
        Raises:
            ValueError: If loan_id already exists in portfolio.
        """
        if loan_id in self.loans:
            raise ValueError(f"Loan {loan_id} already exists in portfolio.")
        
        loan = Loan(
            loan_id=loan_id,
            principal=principal,
            fee=fee,
            duration_days=duration_days,
            target_annual_irr=target_annual_irr,
            irr_function=irr_function
        )
        
        self.loans[loan_id] = loan
        return loan
    
    def process_loan_setup(self, loan_id: str) -> float:
        """
        Finalize a loan setup by determining its revenue share rate.
        
        This must be called after all transactions have been added (or before
        processing live transactions if using the loans with pre-known schedules).
        
        Args:
            loan_id: ID of the loan to set up.
        
        Returns:
            The determined revenue share rate.
        
        Raises:
            KeyError: If loan_id does not exist.
        """
        if loan_id not in self.loans:
            raise KeyError(f"Loan {loan_id} not found in portfolio.")
        
        loan = self.loans[loan_id]
        return loan.determine_revenue_share_rate()
    
    def process_txn(
        self,
        loan_id: str,
        day: int,
        txn_id: str,
        gross_amount: float
    ) -> Dict[str, any]:
        """
        Process a transaction for a specific loan.
        
        Args:
            loan_id: ID of the loan receiving the transaction.
            day: Day on which transaction occurs.
            txn_id: Transaction identifier.
            gross_amount: Gross transaction amount.
        
        Returns:
            Transaction result dict (see Loan.process_txn).
        
        Raises:
            KeyError: If loan_id does not exist.
        """
        if loan_id not in self.loans:
            raise KeyError(f"Loan {loan_id} not found in portfolio.")
        
        return self.loans[loan_id].process_txn(day, txn_id, gross_amount)
    
    def get_loan(self, loan_id: str) -> Loan:
        """
        Retrieve a loan by ID.
        
        Args:
            loan_id: ID of the loan.
        
        Returns:
            The Loan object.
        
        Raises:
            KeyError: If loan_id does not exist.
        """
        if loan_id not in self.loans:
            raise KeyError(f"Loan {loan_id} not found in portfolio.")
        return self.loans[loan_id]
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """
        Get a summary of the entire portfolio.
        
        Returns:
            Dict with counts and status information.
        """
        total_principal = sum(loan.principal for loan in self.loans.values())
        total_fees = sum(loan.fee for loan in self.loans.values())
        total_target_repay = sum(loan.target_repayment for loan in self.loans.values())
        
        status_counts = defaultdict(int)
        for loan in self.loans.values():
            status_counts[loan.loan_status] += 1
        
        return {
            'num_loans': len(self.loans),
            'total_principal': total_principal,
            'total_fees': total_fees,
            'total_target_repayment': total_target_repay,
            'status_breakdown': dict(status_counts),
            'loan_ids': list(self.loans.keys())
        }


# Example IRR calculation function
def calculate_annualized_irr(daily_cash_flows: List[float]) -> float:
    """
    Calculate annualized IRR from a sequence of daily cash flows.
    
    Assumes daily compounding and annualizes over a 365-day year.
    Uses the formula: IRR_annual = (1 + daily_rate)^365 - 1
    
    This is a simplified implementation. In production, use a proper
    financial library or implement Newton-Raphson for robustness.
    
    Args:
        daily_cash_flows: List of daily cash flows (index is day number).
                         Day 0 should be the initial investment (negative).
    
    Returns:
        Annualized IRR as a decimal (e.g., 0.10 for 10%).
    """
    if not daily_cash_flows or daily_cash_flows[0] >= 0:
        raise ValueError("Cash flows must start with a negative outflow.")
    
    # Try NPV at various rates to estimate IRR
    def npv(rate, flows):
        """Net present value at a given daily rate."""
        if rate <= -1.0 or rate == 0:
            return float('inf')  # Avoid division by zero or underflow
        total = 0.0
        for i, flow in enumerate(flows):
            denominator = (1 + rate) ** i
            if denominator == 0:  # Catch underflow to zero
                return float('inf')
            total += flow / denominator
        return total
    
    # Find the daily rate that makes NPV = 0
    try:
        daily_rate = brentq(npv, -0.9999, 1.0, args=(daily_cash_flows,))
        annual_rate = (1 + daily_rate) ** 365 - 1
        return annual_rate
    except ValueError:
        # If no IRR exists, return a large negative or positive value
        return float('-inf')


# ============================================================================
# Usage Examples
# ============================================================================

def simple_test():
    """Basic test case with a single loan."""
    print("=" * 70)
    print("SIMPLE TEST: Single Loan with Constant Daily Revenue")
    print("=" * 70)
    
    manager = LoanManager()
    
    # Create a sample loan
    loan_id = "LOAN_001"
    principal = 1000.0
    fee = 100.0
    duration_days = 365
    target_irr = 0.10  # 10% annualized
    
    loan = manager.create_loan(
        loan_id=loan_id,
        principal=principal,
        fee=fee,
        duration_days=duration_days,
        target_annual_irr=target_irr,
        irr_function=calculate_annualized_irr
    )
    
    # Add transactions: assume daily revenue
    for day in range(1, 366):
        daily_revenue = 3.0  # $3 per day
        loan.add_transaction(day, f"TXN_{day}", daily_revenue)
    
    # Determine the revenue share rate
    rate = manager.process_loan_setup(loan_id)
    print(f"Determined revenue share rate: {rate:.2%}")
    print(f"Target IRR: {target_irr:.2%}")
    print(f"Achieved IRR: {loan.compute_irr(rate):.2%}")
    
    # Process a sample transaction
    result = manager.process_txn(
        loan_id=loan_id,
        day=100,
        txn_id="SAMPLE_TXN",
        gross_amount=50.0
    )
    print(f"\nTransaction result: {result}")
    
    # Get portfolio summary
    summary = manager.get_portfolio_summary()
    print(f"\nPortfolio summary: {summary}\n")


def complex_test():
    """Complex test case with multiple loans and varying revenue patterns."""
    print("=" * 70)
    print("COMPLEX TEST: Multi-Loan Portfolio with Varying Revenue Patterns")
    print("=" * 70)
    
    manager = LoanManager()
    
    # Loan 1: Early repayment scenario (high daily revenue)
    loan1_id = "LOAN_EARLY"
    loan1 = manager.create_loan(
        loan_id=loan1_id,
        principal=5000.0,
        fee=500.0,
        duration_days=180,
        target_annual_irr=0.15,  # 15% IRR
        irr_function=calculate_annualized_irr
    )
    
    # High daily revenue - will repay early
    for day in range(1, 181):
        daily_revenue = 50.0  # $50 per day
        loan1.add_transaction(day, f"LOAN_EARLY_TXN_{day}", daily_revenue)
    
    # Loan 2: Low revenue scenario (requires longer repayment)
    loan2_id = "LOAN_SLOW"
    loan2 = manager.create_loan(
        loan_id=loan2_id,
        principal=2000.0,
        fee=200.0,
        duration_days=365,
        target_annual_irr=0.08,  # 8% IRR (lower given slow growth)
        irr_function=calculate_annualized_irr
    )
    
    # Varying daily revenue - ramps up over time (growing business)
    for day in range(1, 366):
        # Revenue grows gradually: base 5 + (day/365) * 5
        daily_revenue = 5.0 + (day / 365.0) * 5.0
        loan2.add_transaction(day, f"LOAN_SLOW_TXN_{day}", daily_revenue)
    
    # Loan 3: High-risk short-term loan
    loan3_id = "LOAN_SHORT"
    loan3 = manager.create_loan(
        loan_id=loan3_id,
        principal=1000.0,
        fee=250.0,
        duration_days=90,
        target_annual_irr=0.25,  # 25% IRR (higher risk)
        irr_function=calculate_annualized_irr
    )
    
    # High volatility: spiky revenue pattern
    for day in range(1, 91):
        if day % 7 == 0:  # Weekend boost
            daily_revenue = 100.0
        else:
            daily_revenue = 30.0
        loan3.add_transaction(day, f"LOAN_SHORT_TXN_{day}", daily_revenue)
    
    # Set up all loans
    print("\nSetting up loans...")
    for loan_id in [loan1_id, loan2_id, loan3_id]:
        rate = manager.process_loan_setup(loan_id)
        loan = manager.get_loan(loan_id)
        achieved_irr = loan.compute_irr(rate)
        print(f"  {loan_id}:")
        print(f"    Revenue share rate: {rate:.2%}")
        print(f"    Target IRR: {loan.target_annual_irr:.2%}")
        print(f"    Achieved IRR: {achieved_irr:.2%}")
    
    # Process transactions at various days
    print("\n\nProcessing transactions at key days:")
    
    # Day 30 transaction on LOAN_EARLY
    result1 = manager.process_txn(
        loan_id=loan1_id,
        day=30,
        txn_id="DAY30_TXN",
        gross_amount=75.0
    )
    print(f"\n  Day 30 - LOAN_EARLY (active):")
    print(f"    Gross: ${result1['gross_amount']:.2f}")
    print(f"    Lender share: ${result1['lender_share']:.2f}")
    print(f"    Borrower keeps: ${result1['borrower_keeps']:.2f}")
    
    # Day 200 transaction on LOAN_SLOW (post-maturity for LOAN_EARLY but active for LOAN_SLOW)
    result2 = manager.process_txn(
        loan_id=loan2_id,
        day=200,
        txn_id="DAY200_TXN",
        gross_amount=40.0
    )
    print(f"\n  Day 200 - LOAN_SLOW (active):")
    print(f"    Gross: ${result2['gross_amount']:.2f}")
    print(f"    Lender share: ${result2['lender_share']:.2f}")
    print(f"    Borrower keeps: ${result2['borrower_keeps']:.2f}")
    
    # Day 95 transaction on LOAN_SHORT (post-maturity sweep - day 95 > duration 90)
    result3 = manager.process_txn(
        loan_id=loan3_id,
        day=95,
        txn_id="DAY95_TXN",
        gross_amount=60.0
    )
    print(f"\n  Day 95 - LOAN_SHORT (post-maturity, 100% revenue capture):")
    print(f"    Gross: ${result3['gross_amount']:.2f}")
    print(f"    Lender share: ${result3['lender_share']:.2f}")
    print(f"    Borrower keeps: ${result3['borrower_keeps']:.2f}")
    
    # Get portfolio summary
    summary = manager.get_portfolio_summary()
    print(f"\n\nPortfolio Summary:")
    print(f"  Number of loans: {summary['num_loans']}")
    print(f"  Total principal: ${summary['total_principal']:.2f}")
    print(f"  Total fees: ${summary['total_fees']:.2f}")
    print(f"  Total target repayment: ${summary['total_target_repayment']:.2f}")
    print(f"  Status breakdown: {summary['status_breakdown']}")
    print(f"  Loan IDs: {summary['loan_ids']}\n")


def early_repayment_test():
    """Test case where loan is repaid early before maturity."""
    print("=" * 70)
    print("EARLY REPAYMENT TEST: Loan Repaid Before Maturity")
    print("=" * 70)
    
    manager = LoanManager()
    
    # Create a loan with very high daily revenue (will repay quickly)
    loan_id = "LOAN_QUICK"
    principal = 1000.0
    fee = 100.0
    duration_days = 365  # Full year available, but won't need it
    target_irr = 0.20  # 20% IRR
    
    loan = manager.create_loan(
        loan_id=loan_id,
        principal=principal,
        fee=fee,
        duration_days=duration_days,
        target_annual_irr=target_irr,
        irr_function=calculate_annualized_irr
    )
    
    # Very high daily revenue - $200/day
    # Total needed: principal + fee = $1,100
    # Days to repay: 1100 / 200 = 5.5 days (very quick!)
    for day in range(1, 366):
        daily_revenue = 200.0
        loan.add_transaction(day, f"LOAN_QUICK_TXN_{day}", daily_revenue)
    
    # Set up the loan
    print("\nSetting up high-revenue loan...")
    rate = manager.process_loan_setup(loan_id)
    achieved_irr = loan.compute_irr(rate)
    
    print(f"  Principal: ${principal:.2f}")
    print(f"  Fee: ${fee:.2f}")
    print(f"  Target repayment: ${principal + fee:.2f}")
    print(f"  Daily revenue: $200.00")
    print(f"  Revenue share rate: {rate:.2%}")
    print(f"  Target IRR: {target_irr:.2%}")
    print(f"  Achieved IRR: {achieved_irr:.2%}")
    
    # Examine the cash flow structure
    cash_flows = loan.simulate_cash_flows(rate)
    print(f"\n  Cash flows (first 15 days):")
    for day in range(min(15, len(cash_flows))):
        print(f"    Day {day:3d}: ${cash_flows[day]:8.2f}")
    print(f"  Total cash flow days: {len(cash_flows)}")
    print(f"  (Notice: cash flows stop early since loan is fully repaid)")
    
    # Calculate which day the loan is fully repaid
    outstanding = principal + fee
    for day in range(1, len(cash_flows)):
        if cash_flows[day] > 0:
            outstanding -= cash_flows[day]
            if outstanding <= 0:
                print(f"\n  Loan fully repaid on day {day}")
                break
    
    # Process a transaction well into the future (after repayment)
    print("\n\nProcessing transactions after early repayment:")
    
    result1 = manager.process_txn(
        loan_id=loan_id,
        day=7,
        txn_id="POST_REPAY_TXN",
        gross_amount=100.0
    )
    print(f"\n  Day 7 (post-repayment, but still within duration):")
    print(f"    Gross: ${result1['gross_amount']:.2f}")
    print(f"    Lender share: ${result1['lender_share']:.2f}")
    print(f"    Borrower keeps: ${result1['borrower_keeps']:.2f}")
    print(f"    Note: This transaction is processed but shouldn't affect IRR")
    print(f"          (loan is already fully repaid)")
    
    result2 = manager.process_txn(
        loan_id=loan_id,
        day=400,
        txn_id="WAY_FUTURE_TXN",
        gross_amount=150.0
    )
    print(f"\n  Day 400 (post-maturity and post-repayment):")
    print(f"    Gross: ${result2['gross_amount']:.2f}")
    print(f"    Lender share: ${result2['lender_share']:.2f}")
    print(f"    Effective rate: {result2['effective_rate']:.2%}")
    print(f"    Loan status: {result2['loan_status']}")
    print(f"    Note: Even after post-maturity sweep, no additional cash flows")
    print(f"          because loan was already repaid\n")


if __name__ == "__main__":
    # simple_test()
    # complex_test()
    early_repayment_test()
