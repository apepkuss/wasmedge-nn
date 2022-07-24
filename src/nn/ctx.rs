use crate::error::{UsageError, WasiNnError};
use crate::nn::{
    backend::{openvino::OpenvinoBackend, Backend, BackendExecutionContext, BackendGraph},
    ExecutionTarget, GraphEncoding, Tensor,
};
use std::{cell::RefCell, collections::HashMap, hash::Hash};

pub type WasiNnResult<T> = std::result::Result<T, WasiNnError>;
pub type Graph = wasi_nn::Graph;
pub type GraphExecutionContext = wasi_nn::GraphExecutionContext;

pub struct WasiNnCtx {
    ctx: RefCell<Ctx>,
}
impl WasiNnCtx {
    /// Create a new `WasiNnCtx` with the default settings.
    pub fn new() -> WasiNnResult<Self> {
        Ok(Self {
            ctx: RefCell::new(Ctx::new()?),
        })
    }

    pub fn load(
        &mut self,
        architecure: &str,
        weights: &str,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> WasiNnResult<Graph> {
        let arch = std::fs::read_to_string(architecure).map_err(|e| WasiNnError::InvalidPath(e))?;
        let arch_bytes = arch.into_bytes();
        println!("Load graph XML, size in bytes: {}", arch_bytes.len());

        let weights = std::fs::read(weights).map_err(|e| WasiNnError::InvalidPath(e))?;
        println!("Load graph weights, size in bytes: {}", weights.len());

        self.load_from_bytes(arch_bytes.as_slice(), weights.as_slice(), encoding, target)
    }

    pub fn load_from_bytes(
        &mut self,
        architecure: &[u8],
        weights: &[u8],
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> WasiNnResult<Graph> {
        let encoding_id = encoding.to_string();
        let graph =
            if let Some(backend) = self.ctx.borrow_mut().backends.get_mut(encoding_id.as_str()) {
                backend.load(architecure, weights, target)?
            } else {
                return Err(UsageError::InvalidEncoding(encoding.into()).into());
            };
        let graph_id = self.ctx.borrow_mut().graphs.insert(graph);
        Ok(graph_id)
    }

    pub fn init_execution_context(&mut self, graph: Graph) -> WasiNnResult<GraphExecutionContext> {
        let exec_context = if let Some(graph) = self.ctx.borrow_mut().graphs.get_mut(graph) {
            graph.init_execution_context()?
        } else {
            return Err(UsageError::InvalidGraphHandle.into());
        };

        let exec_context_id = self.ctx.borrow_mut().executions.insert(exec_context);
        Ok(exec_context_id)
    }

    pub fn set_input(
        &mut self,
        exec_ctx: GraphExecutionContext,
        index: u32,
        tensor: Tensor,
    ) -> WasiNnResult<()> {
        if let Some(exec_context) = self.ctx.borrow_mut().executions.get_mut(exec_ctx) {
            Ok(exec_context.set_input(index, tensor)?)
        } else {
            Err(UsageError::InvalidGraphHandle.into())
        }
    }

    pub fn compute(&mut self, exec_ctx: GraphExecutionContext) -> WasiNnResult<()> {
        if let Some(exec_context) = self.ctx.borrow_mut().executions.get_mut(exec_ctx) {
            Ok(exec_context.compute()?)
        } else {
            Err(UsageError::InvalidExecutionContextHandle.into())
        }
    }

    pub fn get_output(
        &mut self,
        exec_ctx: GraphExecutionContext,
        index: u32,
        out_buffer: &mut [u8],
    ) -> WasiNnResult<u32> {
        if let Some(exec_context) = self.ctx.borrow_mut().executions.get_mut(exec_ctx) {
            Ok(exec_context.get_output(index, out_buffer)?)
        } else {
            Err(UsageError::InvalidGraphHandle.into())
        }
    }
}

pub(crate) struct Ctx {
    pub(crate) backends: HashMap<String, Box<dyn Backend>>,
    pub(crate) graphs: Table<wasi_nn::Graph, Box<dyn BackendGraph>>,
    pub(crate) executions: Table<wasi_nn::GraphExecutionContext, Box<dyn BackendExecutionContext>>,
}
impl Ctx {
    pub fn new() -> WasiNnResult<Self> {
        let mut backends = HashMap::new();
        backends.insert(
            String::from("openvino"),
            Box::new(OpenvinoBackend::default()) as Box<dyn Backend>,
        );
        Ok(Self {
            backends,
            graphs: Table::default(),
            executions: Table::default(),
        })
    }
}

/// Record handle entries in a table.
pub struct Table<K, V> {
    entries: HashMap<K, V>,
    next_key: u32,
}
impl<K, V> Default for Table<K, V> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            next_key: 0,
        }
    }
}
impl<K, V> Table<K, V>
where
    K: Eq + Hash + From<u32> + Copy,
{
    pub fn insert(&mut self, value: V) -> K {
        let key = self.use_next_key();
        self.entries.insert(key, value);
        key
    }

    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.entries.get_mut(&key)
    }

    fn use_next_key(&mut self) -> K {
        let current = self.next_key;
        self.next_key += 1;
        K::from(current)
    }
}
