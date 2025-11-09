//! RISCV DTM based on the ARM Debug Interface (ADI)
//!
//! This is used in mixed architecture chips.
//!

use crate::{
    MemoryInterface,
    architecture::{
        arm::{
            ArmError,
            ap::{AddressIncrement, ApRegister, CSW, DRW, TAR},
            memory::ArmMemoryInterface,
        },
        riscv::{
            communication_interface::{
                RiscvCommunicationInterface, RiscvDebugInterfaceState, RiscvError,
                RiscvInterfaceBuilder,
            },
            dtm::dtm_access::{DmAddress, DtmAccess},
        },
    },
    probe::{CommandQueue, CommandResult, DebugProbeError, DeferredResultIndex, DeferredResultSet},
};

#[derive(Default)]
pub struct DtmState {
    queued_cmds: CommandQueue<Command>,
    result_set: DeferredResultSet<CommandResult>,
    offset: u64,
}

// TODO: Should this be the ArmDebugInterface?
pub struct AdiDtmBuilder<'probe> {
    probe: Box<dyn ArmMemoryInterface + 'probe>,
    offset: Option<u64>,
}

impl<'probe> AdiDtmBuilder<'probe> {
    pub fn new(probe: Box<dyn ArmMemoryInterface + 'probe>, offset: Option<u64>) -> Self {
        Self { probe, offset }
    }
}

impl<'probe> RiscvInterfaceBuilder<'probe> for AdiDtmBuilder<'probe> {
    fn create_state(&self) -> RiscvDebugInterfaceState {
        let mut state = DtmState::default();
        state.offset = self.offset.unwrap_or(0);
        RiscvDebugInterfaceState::new(Box::new(state))
    }

    fn attach<'state>(
        self: Box<Self>,
        state: &'state mut RiscvDebugInterfaceState,
    ) -> Result<RiscvCommunicationInterface<'state>, DebugProbeError>
    where
        'probe: 'state,
    {
        let dtm_state = state.dtm_state.downcast_mut::<DtmState>().unwrap();
        Ok(RiscvCommunicationInterface::new(
            Box::new(AdiDtm::new(self.probe, dtm_state)),
            &mut state.interface_state,
        ))
    }
}

enum Command {
    Read(DmAddress),
    Write(DmAddress, u32),
}

impl Command {
    /// Map DmAddress to ArmMemoryInterface address.
    /// The ArmMemoryInterface is byte addressed, while the DmAddress is word addressed.
    fn map_addr(&self, offset: u64) -> u32 {
        let address = match self {
            Command::Read(address) => address,
            Command::Write(address, _) => address,
        };
        u32::from(address.0 * 4) + offset as u32
    }
}

/// Access to the Debug Transport Module (DTM),
/// which is used to communicate with the RISC-V debug module.
pub struct AdiDtm<'probe> {
    pub probe: Box<dyn ArmMemoryInterface + 'probe>,
    state: &'probe mut DtmState,
}

impl<'probe> AdiDtm<'probe> {
    pub fn new(probe: Box<dyn ArmMemoryInterface + 'probe>, state: &'probe mut DtmState) -> Self {
        Self { probe, state }
    }

    /// Map DmAddress to ArmMemoryInterface address.
    /// The ArmMemoryInterface is byte addressed, while the DmAddress is word addressed.
    pub fn map_addr(&self, address: DmAddress) -> u64 {
        u64::from(address.0 * 4) + self.state.offset
    }
}

impl DtmAccess for AdiDtm<'_> {
    fn target_reset_assert(&mut self) -> Result<(), DebugProbeError> {
        todo!()
    }

    fn target_reset_deassert(&mut self) -> Result<(), DebugProbeError> {
        todo!()
    }

    fn clear_error_state(&mut self) -> Result<(), RiscvError> {
        // TODO: The ARM debug interface has an error state,
        // maybe this should be a reconnect?
        Ok(())
    }

    fn read_deferred_result(
        &mut self,
        index: DeferredResultIndex,
    ) -> Result<CommandResult, RiscvError> {
        let result = match self.state.result_set.take(index) {
            Ok(value) => value,
            Err(index) => {
                self.execute()?;
                let value = self
                    .state
                    .result_set
                    .take(index)
                    .map_err(|_| RiscvError::BatchedResultNotAvailable)?;
                value
            }
        };
        Ok(result)
    }

    fn execute(&mut self) -> Result<(), RiscvError> {
        let ap = self.probe.fully_qualified_address();
        let offset = self.state.offset;
        let interface = self.probe.get_arm_debug_interface()?;

        // Disable address auto increment
        let csw = interface
            .read_raw_ap_register(&ap, CSW::ADDRESS)
            .map_err(|_| RiscvError::DtmOperationFailed)?
            .try_into()
            .map_err(|_| RiscvError::DtmOperationFailed)?;
        let new_csw = CSW {
            AddrInc: AddressIncrement::Off,
            ..csw
        };
        interface
            .write_raw_ap_register(&ap, CSW::ADDRESS, new_csw.into())
            .map_err(|_| RiscvError::DtmOperationFailed)?;

        let cmds = std::mem::take(&mut self.state.queued_cmds);
        let mut previous_mapped = u32::MAX;
        for (index, cmd) in cmds.iter() {
            let mapped = cmd.map_addr(offset);
            if previous_mapped != mapped {
                interface
                    .write_raw_ap_register(&ap, TAR::ADDRESS, mapped)
                    .map_err(|_| RiscvError::DtmOperationFailed)?;
                previous_mapped = mapped;
            }
            match cmd {
                Command::Read(address) => {
                    let value = interface
                        .read_raw_ap_register(&ap, DRW::ADDRESS)
                        .map_err(|e| RiscvError::DmReadFailed {
                            address: address.0,
                            source: Box::new(e),
                        })?;
                    self.state.result_set.push(index, CommandResult::U32(value));
                }
                Command::Write(address, value) => {
                    interface
                        .write_raw_ap_register(&ap, DRW::ADDRESS, *value)
                        .map_err(|e| RiscvError::DmWriteFailed {
                            address: address.0,
                            source: Box::new(e),
                        })?;
                }
            }
        }

        // Restore CSW
        interface
            .write_raw_ap_register(&ap, CSW::ADDRESS, csw.into())
            .map_err(|_| RiscvError::DtmOperationFailed)?;

        Ok(())
    }

    fn schedule_write(
        &mut self,
        address: DmAddress,
        value: u32,
    ) -> Result<Option<DeferredResultIndex>, RiscvError> {
        self.state
            .queued_cmds
            .schedule(Command::Write(address, value));
        Ok(None)
    }

    fn schedule_read(&mut self, address: DmAddress) -> Result<DeferredResultIndex, RiscvError> {
        Ok(self.state.queued_cmds.schedule(Command::Read(address)))
    }

    fn read_with_timeout(
        &mut self,
        address: DmAddress,
        _timeout: std::time::Duration,
    ) -> Result<u32, RiscvError> {
        Ok(self
            .probe
            .read_word_32(self.map_addr(address))
            .map_err(|e| RiscvError::DmReadFailed {
                address: address.0,
                source: Box::new(e),
            })?)
    }

    fn write_with_timeout(
        &mut self,
        address: DmAddress,
        value: u32,
        _timeout: std::time::Duration,
    ) -> Result<Option<u32>, RiscvError> {
        self.probe
            .write_word_32(self.map_addr(address), value)
            .map_err(|e| RiscvError::DmWriteFailed {
                address: address.0,
                source: Box::new(e),
            })?;
        Ok(None)
    }

    fn read_idcode(&mut self) -> Result<Option<u32>, DebugProbeError> {
        Ok(None)
    }

    fn memory_interface<'m>(
        &'m mut self,
    ) -> Result<&'m mut dyn MemoryInterface<ArmError>, DebugProbeError> {
        Ok(self.probe.as_mut())
    }
}
